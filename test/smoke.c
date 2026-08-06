/*
 * Smoke test for the C API: compiles the real header as C11 and links the real
 * library, so `zig build test` proves the header is valid C and the ABI works
 * end to end. Run by the `test` step; prints nothing and exits 0 on success.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "crv.h"

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,   \
                    #cond);                                                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(void) {
    /* The IR is a value the caller owns; nothing here allocates it. */
    struct crv_ir ir;
    crv_ir_init(&ir);
    CHECK(crv_version_string() != NULL);

    /* rand bit [3:0] x;  constraint c { x inside {[3:7]}; } */
    crv_var x;
    crv_node xr, lo, hi, rng, member;
    CHECK(crv_ir_reserve(&ir, 8, 1, 8) == CRV_OK);
    CHECK(crv_var_add(&ir, 1, 4, CRV_VAR_RAND, &x) == CRV_OK);
    CHECK(crv_node_var(&ir, x, &xr) == CRV_OK);
    CHECK(crv_node_const_u64(&ir, 3, 4, &lo) == CRV_OK);
    CHECK(crv_node_const_u64(&ir, 7, 4, &hi) == CRV_OK);
    CHECK(crv_node_range(&ir, lo, hi, &rng) == CRV_OK);
    CHECK(crv_node_in(&ir, xr, &rng, 1, &member) == CRV_OK);
    CHECK(crv_constraint_add(&ir, 2, 0, &member, 1, NULL) == CRV_OK);
    CHECK(crv_ir_validate(&ir) == CRV_OK);

    /* A membership test is a boolean, whatever it tests. */
    CHECK(crv_node_width(&ir, member) == 1);

    /* Widths only change through a cast, so a cast is how two operands are
     * made to agree. (Adding `xr` and `byte` directly is a precondition
     * violation, not a status: a checked build would panic here.) */
    crv_node byte, widened, sum;
    CHECK(crv_node_const_u64(&ir, 1, 8, &byte) == CRV_OK);
    CHECK(crv_node_cast(&ir, CRV_CAST_ZEXT, xr, 8, &widened) == CRV_OK);
    CHECK(crv_node_width(&ir, widened) == 8);
    CHECK(crv_node_binary(&ir, CRV_OP_ADD, widened, byte, &sum) == CRV_OK);
    CHECK(crv_node_width(&ir, sum) == 8);

    /* Solve. */
    crv_rejection_options options;
    memset(&options, 0, sizeof options);
    options.seed = 0x1234;

    struct crv_solver *solver = NULL;
    CHECK(crv_rejection_sampler_new(&ir, &options, &solver) == CRV_OK);
    CHECK(solver != NULL);

    size_t words = crv_value_words(&ir);
    CHECK(words == 1);

    {
        uint64_t values[1];
        int i;
        for (i = 0; i < 100; i++) {
            CHECK(crv_solver_next(solver, values, words) == CRV_OK);
            CHECK(values[0] >= 3 && values[0] <= 7);
        }

        crv_stats stats;
        crv_solver_stats(solver, &stats);
        CHECK(stats.hits == 100);
        CHECK(stats.attempts >= stats.hits);
    }
    /* The IR stays frozen until the solver borrowing it is gone. */
    crv_solver_free(solver);

    /* Hash, serialize, reload, and confirm the reload is the same query. */
    {
        uint8_t before[CRV_DIGEST_LEN], after[CRV_DIGEST_LEN];
        size_t size;
        unsigned char *blob;
        struct crv_ir loaded;
        crv_var_info info;

        crv_ir_hash(&ir, before);

        size = crv_ir_serialized_size(&ir);
        CHECK(size > 0);
        blob = (unsigned char *)malloc(size);
        CHECK(blob != NULL);
        CHECK(crv_ir_serialize(&ir, blob, size) == CRV_OK);

        CHECK(crv_ir_deserialize(blob, size, &loaded) == CRV_OK);
        crv_ir_hash(&loaded, after);
        CHECK(memcmp(before, after, CRV_DIGEST_LEN) == 0);
        CHECK(crv_var_count(&loaded) == 1);
        CHECK(crv_constraint_count(&loaded) == 1);
        crv_var_get(&loaded, 0, &info);
        CHECK(info.id == 1);
        CHECK(info.width == 4);
        CHECK(info.kind == CRV_VAR_RAND);
        crv_ir_deinit(&loaded);

        /* A damaged blob is a status, not a crash, and leaves storage that is
         * still safe to hand back. */
        blob[size / 2] ^= 0xff;
        CHECK(crv_ir_deserialize(blob, size, &loaded) ==
              CRV_ERR_CHECKSUM_MISMATCH);
        CHECK(crv_ir_node_count(&loaded) == 0);
        crv_ir_deinit(&loaded);
        free(blob);
    }

    /* A node this engine cannot evaluate is refused up front rather than
     * aborting the process partway through a draw. */
    {
        crv_node refs[2], distinct;
        struct crv_solver *unsupported = NULL;
        refs[0] = xr;
        refs[1] = widened;
        CHECK(crv_node_unique(&ir, refs, 2, &distinct) == CRV_OK);
        CHECK(crv_constraint_add(&ir, 3, CRV_CONSTRAINT_SOFT, &distinct, 1,
                                 NULL) == CRV_OK);
        CHECK(crv_ir_validate(&ir) == CRV_OK);
        CHECK(crv_rejection_sampler_new(&ir, NULL, &unsupported) ==
              CRV_ERR_UNSUPPORTED_NODE);
        CHECK(unsupported == NULL);
    }

    crv_ir_deinit(&ir);
    return 0;
}
