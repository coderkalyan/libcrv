//! Bitwuzla backend — the real one.
//!
//! A thin, typed layer over Bitwuzla's C API. Everything above it (the encoder,
//! the sampler) is written against *this* surface rather than against the C
//! header, so that code compiles identically whether or not the backend is
//! linked; `disabled.zig` mirrors this file's public API exactly and
//! `build.zig` picks one. That is also why `Kind` is our own enum rather than a
//! re-export of `BitwuzlaKind`: it keeps the header out of every caller.
//!
//! Handles are opaque pointers owned by the `TermManager`. Bitwuzla's
//! reference counting (`bitwuzla_term_copy`/`_release`) is optional and we do
//! not use it — deleting the term manager releases everything at once, which
//! is exactly the lifetime we want.

const std = @import("std");
const Limb = std.math.big.Limb;
const limb_bits = @bitSizeOf(Limb);

const c = @cImport({
    @cInclude("bitwuzla/c/bitwuzla.h");
});

pub const available = true;

pub const Sort = c.BitwuzlaSort;
pub const Term = c.BitwuzlaTerm;

pub const Result = enum { sat, unsat, unknown };

pub const SatSolver = enum { cadical, cms, kissat };
pub const BvSolver = enum { bitblast, prop, preprop };

/// The subset of `BitwuzlaKind` the encoder needs. Boolean-sorted and
/// bit-vector-sorted operators are deliberately distinct here, because
/// Bitwuzla separates the `Bool` and `BV(1)` sorts and conflating them is the
/// easiest way to build a term that silently means something else.
pub const Kind = enum {
    // Boolean
    not,
    and_,
    or_,
    implies,
    iff,
    equal,
    distinct,
    ite,
    // Bit-vector, value-producing
    bv_not,
    bv_neg,
    bv_add,
    bv_sub,
    bv_mul,
    bv_sdiv,
    bv_udiv,
    bv_srem,
    bv_urem,
    bv_and,
    bv_or,
    bv_xor,
    bv_shl,
    bv_shr,
    bv_ashr,
    bv_concat,
    bv_redxor,
    // Bit-vector, predicate-producing (Bool-sorted results)
    bv_ult,
    bv_ule,
    bv_ugt,
    bv_uge,
    bv_slt,
    bv_sle,
    bv_sgt,
    bv_sge,
};

fn ckind(k: Kind) c.BitwuzlaKind {
    return @intCast(switch (k) {
        .not => c.BITWUZLA_KIND_NOT,
        .and_ => c.BITWUZLA_KIND_AND,
        .or_ => c.BITWUZLA_KIND_OR,
        .implies => c.BITWUZLA_KIND_IMPLIES,
        .iff => c.BITWUZLA_KIND_IFF,
        .equal => c.BITWUZLA_KIND_EQUAL,
        .distinct => c.BITWUZLA_KIND_DISTINCT,
        .ite => c.BITWUZLA_KIND_ITE,
        .bv_not => c.BITWUZLA_KIND_BV_NOT,
        .bv_neg => c.BITWUZLA_KIND_BV_NEG,
        .bv_add => c.BITWUZLA_KIND_BV_ADD,
        .bv_sub => c.BITWUZLA_KIND_BV_SUB,
        .bv_mul => c.BITWUZLA_KIND_BV_MUL,
        .bv_sdiv => c.BITWUZLA_KIND_BV_SDIV,
        .bv_udiv => c.BITWUZLA_KIND_BV_UDIV,
        .bv_srem => c.BITWUZLA_KIND_BV_SREM,
        .bv_urem => c.BITWUZLA_KIND_BV_UREM,
        .bv_and => c.BITWUZLA_KIND_BV_AND,
        .bv_or => c.BITWUZLA_KIND_BV_OR,
        .bv_xor => c.BITWUZLA_KIND_BV_XOR,
        .bv_shl => c.BITWUZLA_KIND_BV_SHL,
        .bv_shr => c.BITWUZLA_KIND_BV_SHR,
        .bv_ashr => c.BITWUZLA_KIND_BV_ASHR,
        .bv_concat => c.BITWUZLA_KIND_BV_CONCAT,
        .bv_redxor => c.BITWUZLA_KIND_BV_REDXOR,
        .bv_ult => c.BITWUZLA_KIND_BV_ULT,
        .bv_ule => c.BITWUZLA_KIND_BV_ULE,
        .bv_ugt => c.BITWUZLA_KIND_BV_UGT,
        .bv_uge => c.BITWUZLA_KIND_BV_UGE,
        .bv_slt => c.BITWUZLA_KIND_BV_SLT,
        .bv_sle => c.BITWUZLA_KIND_BV_SLE,
        .bv_sgt => c.BITWUZLA_KIND_BV_SGT,
        .bv_sge => c.BITWUZLA_KIND_BV_SGE,
    });
}

// -- Options -----------------------------------------------------------------

pub const Options = struct {
    ptr: ?*c.BitwuzlaOptions,

    pub fn init() Options {
        return .{ .ptr = c.bitwuzla_options_new() };
    }

    pub fn deinit(o: Options) void {
        c.bitwuzla_options_delete(o.ptr);
    }

    pub fn setSeed(o: Options, seed: u32) void {
        c.bitwuzla_set_option(o.ptr, @intCast(c.BITWUZLA_OPT_SEED), seed);
    }

    pub fn setProduceModels(o: Options, on: bool) void {
        c.bitwuzla_set_option(o.ptr, @intCast(c.BITWUZLA_OPT_PRODUCE_MODELS), @intFromBool(on));
    }

    pub fn setProduceUnsatAssumptions(o: Options, on: bool) void {
        c.bitwuzla_set_option(
            o.ptr,
            @intCast(c.BITWUZLA_OPT_PRODUCE_UNSAT_ASSUMPTIONS),
            @intFromBool(on),
        );
    }

    /// Note that a mode Bitwuzla was not built with is a fatal error inside the
    /// library, not something this call can report. `libbitwuzla` must be built
    /// with CryptoMiniSat for `.cms`.
    pub fn setSatSolver(o: Options, s: SatSolver) void {
        c.bitwuzla_set_option_mode(o.ptr, @intCast(c.BITWUZLA_OPT_SAT_SOLVER), switch (s) {
            .cadical => "cadical",
            .cms => "cms",
            .kissat => "kissat",
        });
    }

    pub fn setBvSolver(o: Options, s: BvSolver) void {
        c.bitwuzla_set_option_mode(o.ptr, @intCast(c.BITWUZLA_OPT_BV_SOLVER), switch (s) {
            .bitblast => "bitblast",
            .prop => "prop",
            .preprop => "preprop",
        });
    }
};

// -- Term manager ------------------------------------------------------------

pub const TermManager = struct {
    ptr: ?*c.BitwuzlaTermManager,

    pub fn init() TermManager {
        return .{ .ptr = c.bitwuzla_term_manager_new() };
    }

    /// Releases every sort and term created through this manager.
    pub fn deinit(tm: TermManager) void {
        c.bitwuzla_term_manager_delete(tm.ptr);
    }

    pub fn boolSort(tm: TermManager) Sort {
        return c.bitwuzla_mk_bool_sort(tm.ptr);
    }

    pub fn bvSort(tm: TermManager, width: u64) Sort {
        return c.bitwuzla_mk_bv_sort(tm.ptr, width);
    }

    /// A free constant of the given sort — the solver assigns it.
    pub fn constant(tm: TermManager, sort: Sort) Term {
        return c.bitwuzla_mk_const(tm.ptr, sort, null);
    }

    pub fn mkTrue(tm: TermManager) Term {
        return c.bitwuzla_mk_true(tm.ptr);
    }

    pub fn mkFalse(tm: TermManager) Term {
        return c.bitwuzla_mk_false(tm.ptr);
    }

    pub fn bvValueU64(tm: TermManager, sort: Sort, value: u64) Term {
        return c.bitwuzla_mk_bv_value_uint64(tm.ptr, sort, value);
    }

    /// A bit-vector literal from a base-2 string of exactly the sort's width.
    pub fn bvValueBin(tm: TermManager, sort: Sort, bits: [:0]const u8) Term {
        return c.bitwuzla_mk_bv_value(tm.ptr, sort, bits.ptr, 2);
    }

    pub fn term1(tm: TermManager, kind: Kind, a: Term) Term {
        return c.bitwuzla_mk_term1(tm.ptr, ckind(kind), a);
    }

    pub fn term2(tm: TermManager, kind: Kind, a: Term, b: Term) Term {
        return c.bitwuzla_mk_term2(tm.ptr, ckind(kind), a, b);
    }

    pub fn term3(tm: TermManager, kind: Kind, a: Term, b: Term, d: Term) Term {
        return c.bitwuzla_mk_term3(tm.ptr, ckind(kind), a, b, d);
    }

    pub fn termN(tm: TermManager, kind: Kind, args: []const Term) Term {
        return c.bitwuzla_mk_term(
            tm.ptr,
            ckind(kind),
            @intCast(args.len),
            @constCast(args.ptr),
        );
    }

    /// Bits `hi` down to `lo`, inclusive.
    pub fn extract(tm: TermManager, a: Term, hi: u64, lo: u64) Term {
        return c.bitwuzla_mk_term1_indexed2(
            tm.ptr,
            @intCast(c.BITWUZLA_KIND_BV_EXTRACT),
            a,
            hi,
            lo,
        );
    }

    /// Widen by `n` zero bits.
    pub fn zeroExtend(tm: TermManager, a: Term, n: u64) Term {
        return c.bitwuzla_mk_term1_indexed1(
            tm.ptr,
            @intCast(c.BITWUZLA_KIND_BV_ZERO_EXTEND),
            a,
            n,
        );
    }

    /// Widen by `n` copies of the sign bit.
    pub fn signExtend(tm: TermManager, a: Term, n: u64) Term {
        return c.bitwuzla_mk_term1_indexed1(
            tm.ptr,
            @intCast(c.BITWUZLA_KIND_BV_SIGN_EXTEND),
            a,
            n,
        );
    }
};

// -- Solver session ----------------------------------------------------------

pub const Session = struct {
    ptr: ?*c.Bitwuzla,

    pub fn init(tm: TermManager, options: Options) Session {
        return .{ .ptr = c.bitwuzla_new(tm.ptr, options.ptr) };
    }

    pub fn deinit(s: Session) void {
        c.bitwuzla_delete(s.ptr);
    }

    pub fn assert(s: Session, t: Term) void {
        c.bitwuzla_assert(s.ptr, t);
    }

    pub fn push(s: Session, levels: u64) void {
        c.bitwuzla_push(s.ptr, levels);
    }

    pub fn pop(s: Session, levels: u64) void {
        c.bitwuzla_pop(s.ptr, levels);
    }

    pub fn checkSat(s: Session) Result {
        return mapResult(c.bitwuzla_check_sat(s.ptr));
    }

    pub fn checkSatAssuming(s: Session, assumptions: []const Term) Result {
        return mapResult(c.bitwuzla_check_sat_assuming(
            s.ptr,
            @intCast(assumptions.len),
            @constCast(assumptions.ptr),
        ));
    }

    /// The value of `t` in the current model. Only valid after a `sat` result
    /// and before the next assertion or `check_sat`.
    pub fn getValue(s: Session, t: Term) Term {
        return c.bitwuzla_get_value(s.ptr, t);
    }

    /// Install a callback polled during solving; a non-zero return aborts the
    /// current check, which then reports `unknown`.
    pub fn setTermination(
        s: Session,
        callback: *const fn (?*anyopaque) callconv(.c) c_int,
        state: ?*anyopaque,
    ) void {
        c.bitwuzla_set_termination_callback(s.ptr, callback, state);
    }
};

fn mapResult(r: c.BitwuzlaResult) Result {
    if (r == c.BITWUZLA_SAT) return .sat;
    if (r == c.BITWUZLA_UNSAT) return .unsat;
    return .unknown;
}

// -- Value extraction --------------------------------------------------------

/// Read a bit-vector value term into little-endian limbs.
///
/// Base 2 rather than 16: Bitwuzla emits exactly one character per bit, so this
/// works at every width, including the ones that are not a multiple of four.
/// The returned pointer is only valid until the next call into the same
/// function, so it is consumed immediately and never stored.
pub fn readBits(term: Term, out: []Limb, width: u16) void {
    @memset(out, 0);
    const raw = c.bitwuzla_term_value_get_str_fmt(term, 2) orelse return;
    const str = std.mem.span(raw);
    if (str.len == 0) return;

    // Most-significant character first. Take the low `width` characters, so an
    // unexpected prefix cannot shift every bit.
    const n = @min(str.len, @as(usize, width));
    const tail = str[str.len - n ..];
    for (tail, 0..) |ch, i| {
        if (ch != '1') continue;
        const bit = n - 1 - i;
        out[bit / limb_bits] |= @as(Limb, 1) << @intCast(bit % limb_bits);
    }
}

/// Read a Bool-sorted value term.
pub fn readBool(term: Term) bool {
    return c.bitwuzla_term_value_get_bool(term);
}
