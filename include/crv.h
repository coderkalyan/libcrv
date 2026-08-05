/*
 * libcrv — a constrained random verification library.
 *
 * This is the C interface to the Zig library: build a constraint set into an
 * IR, hash or cache it, then draw satisfying assignments from it.
 *
 *     crv_ir ir;
 *     crv_ir_init(&ir);
 *
 *     crv_var x;
 *     crv_node xr, lo, hi, rng, member;
 *     crv_var_add(&ir, 1, 4, CRV_VAR_RAND, &x);        // rand bit [3:0] x;
 *     crv_node_var(&ir, x, &xr);
 *     crv_node_const_u64(&ir, 0, 4, &lo);
 *     crv_node_const_u64(&ir, 15, 4, &hi);
 *     crv_node_range(&ir, lo, hi, &rng);               // [0:15]
 *     crv_node_in(&ir, xr, &rng, 1, &member);          // x inside {[0:15]}
 *     crv_constraint_add(&ir, 2, 0, &member, 1, NULL); // constraint c { ... }
 *
 *     crv_solver *s;
 *     crv_rejection_sampler_new(&ir, NULL, &s);
 *
 *     uint64_t values[1];
 *     if (crv_solver_next(s, values, 1) == CRV_OK) { ... values[0] ... }
 *
 *     crv_solver_free(s);
 *     crv_ir_deinit(&ir);
 *
 * Ownership and lifetimes:
 *
 *   - A `crv_ir` is a value you place wherever you like — on the stack, in a
 *     struct, in an arena. The library never allocates it; `crv_ir_init` sets
 *     it up and `crv_ir_deinit` releases the arrays hanging off it.
 *   - A `crv_solver` is an opaque handle the library does allocate, because
 *     its size depends on the engine behind it. Free it with `crv_solver_free`.
 *   - A `crv_solver` borrows its `crv_ir`, which must outlive it and must not
 *     be modified while it does. Adding to an IR a solver is bound to is
 *     undefined behaviour, not a detected error.
 *   - No buffer is ever passed across the library boundary for the caller to
 *     free. Where the library produces bytes, the caller supplies the storage.
 *
 * Threading: a `crv_ir` and a `crv_solver` are each single-threaded, but
 * distinct handles share no state, so one thread per (IR, solver) pair needs
 * no locking.
 */

#ifndef LIBCRV_CRV_H
#define LIBCRV_CRV_H

#include <stddef.h>
#include <stdint.h>

#ifndef __cplusplus
/* `alignas` is a keyword in C++ and in C23, a macro from here in C11. */
#include <stdalign.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CRV_VERSION_MAJOR 0
#define CRV_VERSION_MINOR 1
#define CRV_VERSION_PATCH 0

/* Length of a content digest, in bytes (Blake3-256). */
#define CRV_DIGEST_LEN 32

/* "M.m.p" of the library actually linked in. */
const char *crv_version_string(void);

/* -- Status ---------------------------------------------------------------
 *
 * Negative values are hard errors, so `if (status < 0)` is the error test.
 * `CRV_EXHAUSTED` is a normal outcome of solving, not a failure.
 */
typedef enum crv_status {
    CRV_OK = 0,
    /* The solver gave up within its attempt budget. For an incomplete engine
     * this is NOT a proof that the constraints are unsatisfiable. */
    CRV_EXHAUSTED = 1,

    CRV_ERR_OOM = -1,
    /* An out-of-range index, an unknown enum value, or a count/pointer pair
     * that does not describe a valid array. */
    CRV_ERR_INVALID_ARGUMENT = -2,
    /* Operand widths disagree where the evaluator needs one width. Change
     * width only through `crv_node_cast`. */
    CRV_ERR_WIDTH_MISMATCH = -3,
    CRV_ERR_BUFFER_TOO_SMALL = -4,
    /* The IR is not structurally sound; see `crv_ir_validate`. */
    CRV_ERR_INVALID_IR = -5,
    /* The IR contains a node this solver engine cannot evaluate. */
    CRV_ERR_UNSUPPORTED_NODE = -6,
    CRV_ERR_BAD_MAGIC = -7,
    CRV_ERR_UNSUPPORTED_VERSION = -8,
    CRV_ERR_CHECKSUM_MISMATCH = -9,
    CRV_ERR_TRUNCATED = -10
} crv_status;

/* A short, static description of `status`. Never null. */
const char *crv_status_string(crv_status status);

/* -- Handles --------------------------------------------------------------
 *
 * Nodes, variables and constraints are indices into the IR's own arrays, not
 * pointers, and are only meaningful against the `crv_ir` that produced them.
 * On failure a builder writes `CRV_INVALID` through its out-parameter.
 */

/* An IR, as storage rather than as a pointer: `crv_ir` is exactly as large and
 * as aligned as the library's own representation, so C code can hold one by
 * value without a definition of it. The bytes are private — read or write them
 * and you are on your own. The buffer carries slack deliberately, and the
 * library refuses to compile if its representation ever outgrows it, so the
 * size is checked rather than assumed. */
typedef struct crv_ir {
    alignas(8) unsigned char private_storage[128];
} crv_ir;

typedef struct crv_solver crv_solver;

typedef uint32_t crv_node;
typedef uint32_t crv_var;
typedef uint32_t crv_constraint;

#define CRV_INVALID UINT32_MAX

/* An out-parameter below may be NULL to discard the result, except where the
 * result is the whole point of the call (`crv_ir_deserialize`,
 * `crv_rejection_sampler_new`). A `crv_ir *` or `crv_solver *`, by contrast, is
 * dereferenced unchecked: passing NULL is undefined, as it is for most of the C
 * standard library. `crv_solver_free(NULL)` is the one exception, so that
 * cleaning up after a failed constructor needs no guard. */

/* -- IR lifetime ---------------------------------------------------------- */

/* Set up an empty IR in caller-provided storage. Cannot fail: an empty IR owns
 * nothing yet. Pair with `crv_ir_deinit`, which is safe on any initialized IR
 * whatever happened in between. */
void crv_ir_init(crv_ir *ir);
void crv_ir_deinit(crv_ir *ir);

/* Pre-size the IR's arrays. Purely a performance hint: building works
 * without it, `extra` is the payload pool for variable-arity nodes. */
crv_status crv_ir_reserve(crv_ir *ir, uint32_t nodes, uint32_t vars,
                          uint32_t extra);

/* Check that the IR is structurally sound: operand indices in range and
 * referring backwards (so evaluation, a single forward sweep, is well
 * defined), payload slices inside the pool, widths non-zero and in agreement
 * where the evaluator assumes one width. An IR built through this header
 * always passes, and `crv_ir_deserialize` validates before returning, so this
 * is mostly a self-check. */
crv_status crv_ir_validate(const crv_ir *ir);

uint32_t crv_ir_node_count(const crv_ir *ir);

/* -- Variables ------------------------------------------------------------ */

typedef enum crv_var_kind {
    /* Fixed input the solver may read but not assign. */
    CRV_VAR_STATE = 0,
    /* Randomized on each solve. */
    CRV_VAR_RAND = 1,
    /* Randomized cyclically: every value before any repeats. */
    CRV_VAR_RANDC = 2
} crv_var_kind;

typedef struct crv_var_info {
    uint32_t id;
    uint16_t width;
    uint8_t kind; /* crv_var_kind */
    uint8_t reserved;
} crv_var_info;

/* Declare a variable. `id` is an opaque handle of the caller's choosing — the
 * IR stores no names, so resolve it against your own symbol table. `width` is
 * the bit-vector width (1..65535); signedness is not part of a type, it is
 * chosen per operation. */
crv_status crv_var_add(crv_ir *ir, uint32_t id, uint16_t width,
                       crv_var_kind kind, crv_var *out);

uint32_t crv_var_count(const crv_ir *ir);
crv_status crv_var_get(const crv_ir *ir, crv_var v, crv_var_info *out);

/* -- Leaves --------------------------------------------------------------- */

crv_status crv_node_var(crv_ir *ir, crv_var v, crv_node *out);
crv_status crv_node_bool(crv_ir *ir, int value, crv_node *out);

/* A `width`-bit literal. The value is taken modulo `width` bits. */
crv_status crv_node_const_u64(crv_ir *ir, uint64_t value, uint16_t width,
                              crv_node *out);

/* A literal of any width, from `nwords` little-endian 64-bit words — the same
 * layout solutions come back in. Taken modulo `width` bits, and treated as an
 * unsigned bit pattern, so a negative constant is written as its two's
 * complement (or sign-extended from a narrower literal). */
crv_status crv_node_const_bits(crv_ir *ir, const uint64_t *words, size_t nwords,
                               uint16_t width, crv_node *out);

/* -- Operators ------------------------------------------------------------
 *
 * A type is just a width, so operations that depend on signedness come in a
 * signed (`S`) and an unsigned (`U`) form, and an operator's result has its
 * operands' width, wrapped there. Widths change only through
 * `crv_node_cast` — to add two 4-bit values without wrapping, zero-extend
 * them first.
 *
 * These values are the library's own node tags, drawn from one numbering that
 * also fixes the serialized format, so nothing is translated at the boundary
 * and the gaps between the groups below are room for that numbering to grow.
 */
typedef enum crv_op {
    /* Unary. */
    CRV_OP_NEG = 16,  /* -a          */
    CRV_OP_BNOT = 17, /* ~a          */
    CRV_OP_LNOT = 18, /* !a          */

    /* Binary, result width = operand width. */
    CRV_OP_ADD = 32,
    CRV_OP_SUB = 33,
    CRV_OP_MUL = 34,
    CRV_OP_SDIV = 35,
    CRV_OP_UDIV = 36,
    CRV_OP_SMOD = 37,
    CRV_OP_UMOD = 38,
    CRV_OP_BAND = 39,
    CRV_OP_BOR = 40,
    CRV_OP_BXOR = 41,
    CRV_OP_SLL = 42, /* shift left                       */
    CRV_OP_SRL = 43, /* shift right, zero-filling        */
    CRV_OP_SRA = 44, /* shift right, sign-extending      */

    /* Binary, 1-bit result. */
    CRV_OP_EQ = 64,
    CRV_OP_NE = 65,
    CRV_OP_SLT = 66,
    CRV_OP_ULT = 67,
    CRV_OP_SLE = 68,
    CRV_OP_ULE = 69,
    CRV_OP_SGT = 70,
    CRV_OP_UGT = 71,
    CRV_OP_SGE = 72,
    CRV_OP_UGE = 73,
    CRV_OP_LAND = 74,
    CRV_OP_LOR = 75,
    CRV_OP_IMPLIES = 76, /* a -> b  */
    CRV_OP_IFF = 77      /* a <-> b */
} crv_op;

typedef enum crv_cast {
    CRV_CAST_ZEXT = 96,  /* zero-extend to a wider width       */
    CRV_CAST_SEXT = 97,  /* sign-extend to a wider width       */
    CRV_CAST_TRUNC = 98  /* truncate, keeping the low bits     */
} crv_cast;

/* Arity is checked: a binary op passed to `crv_node_unary` (or the reverse)
 * is `CRV_ERR_INVALID_ARGUMENT`. Arithmetic and comparison operands must have
 * equal widths (`CRV_ERR_WIDTH_MISMATCH`); shifts read the shift amount at its
 * own width, and the logical operators only test for non-zero, so those two
 * groups accept mismatched widths. */
crv_status crv_node_unary(crv_ir *ir, crv_op op, crv_node a, crv_node *out);
crv_status crv_node_binary(crv_ir *ir, crv_op op, crv_node a, crv_node b,
                           crv_node *out);
crv_status crv_node_cast(crv_ir *ir, crv_cast cast, crv_node a, uint16_t width,
                         crv_node *out);

/* The width a node evaluates at. Resolved by walking the node's operand chain,
 * so it costs the depth of the expression, not constant time. */
crv_status crv_node_width(const crv_ir *ir, crv_node n, uint16_t *out_width);

/* -- Sets, distributions, structural constraints -------------------------- */

/* Inclusive range `[lo:hi]`, for use as a member of a set or distribution.
 * Not a boolean on its own. */
crv_status crv_node_range(crv_ir *ir, crv_node lo, crv_node hi, crv_node *out);

/* Set membership — `value inside { members... }`. A member is a value node or
 * a range node, and must have the same width as `value`. */
crv_status crv_node_in(crv_ir *ir, crv_node value, const crv_node *members,
                       size_t n, crv_node *out);

typedef enum crv_dist_kind {
    CRV_DIST_EQ = 0, /* `value := weight`, weight per value        */
    CRV_DIST_DIV = 1 /* `value :/ weight`, weight split over range */
} crv_dist_kind;

/* One weighted item of a distribution; `value` is a value node or a range. */
crv_status crv_node_dist_item(crv_ir *ir, crv_dist_kind kind, crv_node value,
                              crv_node weight, crv_node *out);

/* `value dist { items... }`, each item from `crv_node_dist_item`. */
crv_status crv_node_dist(crv_ir *ir, crv_node value, const crv_node *items,
                         size_t n, crv_node *out);

crv_status crv_node_if(crv_ir *ir, crv_node cond, crv_node then_stmt,
                       crv_node *out);
crv_status crv_node_if_else(crv_ir *ir, crv_node cond, crv_node then_stmt,
                            crv_node else_stmt, crv_node *out);

/* `unique { nodes... }` — the listed values must all differ. */
crv_status crv_node_unique(crv_ir *ir, const crv_node *nodes, size_t n,
                           crv_node *out);

/* `solve before... before after...` — a solve-ordering hint, not a boolean. */
crv_status crv_node_solve_before(crv_ir *ir, const crv_var *before,
                                 size_t nbefore, const crv_var *after,
                                 size_t nafter, crv_node *out);

/* -- Constraints ---------------------------------------------------------- */

/* May be dropped if it conflicts with a hard constraint. */
#define CRV_CONSTRAINT_SOFT 0x1u

typedef struct crv_constraint_info {
    uint32_t id;
    uint32_t flags;
    uint32_t stmt_count;
} crv_constraint_info;

/* A named block of boolean statements that must all hold. Like a variable's,
 * `id` is opaque and caller-assigned. */
crv_status crv_constraint_add(crv_ir *ir, uint32_t id, uint32_t flags,
                              const crv_node *stmts, size_t n,
                              crv_constraint *out);

uint32_t crv_constraint_count(const crv_ir *ir);
crv_status crv_constraint_get(const crv_ir *ir, crv_constraint c,
                              crv_constraint_info *out);

/* -- Hashing and caching --------------------------------------------------
 *
 * The digest is a Blake3 content hash: two structurally identical constraint
 * sets hash equal, so it doubles as a collision-resistant cache key.
 *
 * Serialized form is a versioned, checksummed blob. It is portable between
 * machines of the same endianness — enough for a local build cache.
 */
crv_status crv_ir_hash(const crv_ir *ir, uint8_t out[CRV_DIGEST_LEN]);

/* Exact size of this IR's serialized form. Computed without serializing. */
size_t crv_ir_serialized_size(const crv_ir *ir);

/* Write the serialized form into `buf`. If `cap` is too small, nothing is
 * written and `CRV_ERR_BUFFER_TOO_SMALL` is returned; either way `*written`
 * (when non-NULL) receives the required size. */
crv_status crv_ir_serialize(const crv_ir *ir, void *buf, size_t cap,
                            size_t *written);

/* Rebuild an IR from bytes written by `crv_ir_serialize`, into storage the
 * caller supplies — which must not already hold an initialized IR, since this
 * initializes it in place of `crv_ir_init`. The checksum is verified before
 * any length in the blob is trusted, and the result is validated before it is
 * returned, so a corrupt or hostile blob fails with a status rather than
 * producing an IR that reads out of bounds later. A rejected blob leaves `*out`
 * empty rather than untouched, so it is still safe to `crv_ir_deinit`. */
crv_status crv_ir_deserialize(const void *buf, size_t len, crv_ir *out);

/* -- Solving --------------------------------------------------------------
 *
 * A solution is a vector of little-endian 64-bit words: `crv_value_words(&ir)`
 * words per variable, so variable `i` occupies
 * `values[i * words .. (i + 1) * words]`, and `values[i]` is simply variable
 * `i`'s value in the common case where every variable fits in 64 bits.
 *
 * `crv_solver` is one type for every engine: a future exact/SMT backend is a
 * different constructor, not a different handle type.
 */
size_t crv_value_words(const crv_ir *ir);

typedef struct crv_rejection_options {
    uint64_t seed;
    uint32_t max_attempts; /* 0 selects the default (10000) */
    uint32_t reserved;
} crv_rejection_options;

/* Draw random values for every variable and keep the first draw that
 * satisfies every constraint. Simple and fast per attempt, but incomplete:
 * see `CRV_EXHAUSTED`. Pass NULL for `options` to take the defaults.
 *
 * The IR is validated here, and rejected with `CRV_ERR_UNSUPPORTED_NODE` if it
 * uses a node this engine cannot evaluate (`dist` weighting or the structural
 * constraints, today). It must outlive the solver, and must not be modified
 * while the solver exists. */
crv_status crv_rejection_sampler_new(const crv_ir *ir,
                                     const crv_rejection_options *options,
                                     crv_solver **out);

void crv_solver_free(crv_solver *solver);

/* Draw one satisfying assignment into `values`, which must hold at least
 * `crv_value_words(&ir) * crv_var_count(&ir)` words. Returns `CRV_OK` with the
 * buffer filled, or `CRV_EXHAUSTED` if the engine gave up. */
crv_status crv_solver_next(crv_solver *solver, uint64_t *values, size_t nwords);

typedef struct crv_stats {
    uint64_t attempts; /* draws evaluated across every call to next   */
    uint64_t hits;     /* draws accepted                              */
} crv_stats;

crv_status crv_solver_stats(const crv_solver *solver, crv_stats *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LIBCRV_CRV_H */
