// RUN: cudatile-to-gpu --convert-cuda-tile-to-gpu --verify-diagnostics %s

// The `global` variant of cuda_tile.alloca (the second case in the Ops.td
// mlirExample) marks the returned address as shareable across tile threads.
// The pass models a scalar pointer as an unranked memref<*xT> that carries no
// memory space able to express that sharing, so the conversion intentionally
// bails and the op fails to legalize.

cuda_tile.module @alloca_neg {
  entry @global_alloca() {
    // expected-error @below {{failed to legalize operation 'cuda_tile.alloca'}}
    %0 = alloca num_elem = 64, alignment = 128 global : tile<ptr<f32>>
    return
  }
}
