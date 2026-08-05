//! Bitwuzla backend — the stub.
//!
//! Mirrors `enabled.zig`'s public API exactly so that everything above it
//! compiles unchanged when the backend is not linked. `build.zig` selects this
//! file unless `-Dbitwuzla` is passed, which keeps the default build of libcrv
//! dependency-free.
//!
//! Nothing here is reachable: `SmtSampler.init` checks `available` and returns
//! `error.BackendUnavailable` before constructing anything. The bodies panic
//! rather than returning junk so that a future caller which forgets that check
//! fails loudly instead of computing on garbage.

const std = @import("std");
const Limb = std.math.big.Limb;

pub const available = false;

pub const Sort = usize;
pub const Term = usize;

pub const Result = enum { sat, unsat, unknown };

pub const SatSolver = enum { cadical, cms, kissat };
pub const BvSolver = enum { bitblast, prop, preprop };

pub const Kind = enum {
    not,
    and_,
    or_,
    implies,
    iff,
    equal,
    distinct,
    ite,
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
    bv_ult,
    bv_ule,
    bv_ugt,
    bv_uge,
    bv_slt,
    bv_sle,
    bv_sgt,
    bv_sge,
};

fn unavailable() noreturn {
    @panic("libcrv: the Bitwuzla backend is not linked; rebuild with -Dbitwuzla");
}

pub const Options = struct {
    ptr: ?*anyopaque = null,

    pub fn init() Options {
        unavailable();
    }
    pub fn deinit(o: Options) void {
        _ = o;
        unavailable();
    }
    pub fn setSeed(o: Options, seed: u32) void {
        _ = .{ o, seed };
        unavailable();
    }
    pub fn setProduceModels(o: Options, on: bool) void {
        _ = .{ o, on };
        unavailable();
    }
    pub fn setProduceUnsatAssumptions(o: Options, on: bool) void {
        _ = .{ o, on };
        unavailable();
    }
    pub fn setSatSolver(o: Options, s: SatSolver) void {
        _ = .{ o, s };
        unavailable();
    }
    pub fn setBvSolver(o: Options, s: BvSolver) void {
        _ = .{ o, s };
        unavailable();
    }
};

pub const TermManager = struct {
    ptr: ?*anyopaque = null,

    pub fn init() TermManager {
        unavailable();
    }
    pub fn deinit(tm: TermManager) void {
        _ = tm;
        unavailable();
    }
    pub fn boolSort(tm: TermManager) Sort {
        _ = tm;
        unavailable();
    }
    pub fn bvSort(tm: TermManager, width: u64) Sort {
        _ = .{ tm, width };
        unavailable();
    }
    pub fn constant(tm: TermManager, sort: Sort) Term {
        _ = .{ tm, sort };
        unavailable();
    }
    pub fn mkTrue(tm: TermManager) Term {
        _ = tm;
        unavailable();
    }
    pub fn mkFalse(tm: TermManager) Term {
        _ = tm;
        unavailable();
    }
    pub fn bvValueU64(tm: TermManager, sort: Sort, value: u64) Term {
        _ = .{ tm, sort, value };
        unavailable();
    }
    pub fn bvValueBin(tm: TermManager, sort: Sort, bits: [:0]const u8) Term {
        _ = .{ tm, sort, bits };
        unavailable();
    }
    pub fn term1(tm: TermManager, kind: Kind, a: Term) Term {
        _ = .{ tm, kind, a };
        unavailable();
    }
    pub fn term2(tm: TermManager, kind: Kind, a: Term, b: Term) Term {
        _ = .{ tm, kind, a, b };
        unavailable();
    }
    pub fn term3(tm: TermManager, kind: Kind, a: Term, b: Term, d: Term) Term {
        _ = .{ tm, kind, a, b, d };
        unavailable();
    }
    pub fn termN(tm: TermManager, kind: Kind, args: []const Term) Term {
        _ = .{ tm, kind, args };
        unavailable();
    }
    pub fn extract(tm: TermManager, a: Term, hi: u64, lo: u64) Term {
        _ = .{ tm, a, hi, lo };
        unavailable();
    }
    pub fn zeroExtend(tm: TermManager, a: Term, n: u64) Term {
        _ = .{ tm, a, n };
        unavailable();
    }
    pub fn signExtend(tm: TermManager, a: Term, n: u64) Term {
        _ = .{ tm, a, n };
        unavailable();
    }
};

pub const Session = struct {
    ptr: ?*anyopaque = null,

    pub fn init(tm: TermManager, options: Options) Session {
        _ = .{ tm, options };
        unavailable();
    }
    pub fn deinit(s: Session) void {
        _ = s;
        unavailable();
    }
    pub fn assert(s: Session, t: Term) void {
        _ = .{ s, t };
        unavailable();
    }
    pub fn push(s: Session, levels: u64) void {
        _ = .{ s, levels };
        unavailable();
    }
    pub fn pop(s: Session, levels: u64) void {
        _ = .{ s, levels };
        unavailable();
    }
    pub fn checkSat(s: Session) Result {
        _ = s;
        unavailable();
    }
    pub fn checkSatAssuming(s: Session, assumptions: []const Term) Result {
        _ = .{ s, assumptions };
        unavailable();
    }
    pub fn getValue(s: Session, t: Term) Term {
        _ = .{ s, t };
        unavailable();
    }
    pub fn setTermination(
        s: Session,
        callback: *const fn (?*anyopaque) callconv(.c) c_int,
        state: ?*anyopaque,
    ) void {
        _ = .{ s, callback, state };
        unavailable();
    }
};

pub fn readBits(term: Term, out: []Limb, width: u16) void {
    _ = .{ term, out, width };
    unavailable();
}

pub fn readBool(term: Term) bool {
    _ = term;
    unavailable();
}
