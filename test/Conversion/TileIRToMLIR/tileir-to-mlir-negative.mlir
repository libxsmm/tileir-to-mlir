// RUN: tileir-to-mlir --convert-tileir-to-mlir --verify-diagnostics --split-input-file %s

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

// -----

// A ranked pointer tile lowers to per-lane offsets from an implicit base, so it
// cannot be reshaped back into a scalar pointer. The conversion must fail
// rather than reinterpret an offset as a pointer.

cuda_tile.module @ranked_to_scalar_ptr_neg {
  entry @reshape_to_scalar(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %off = constant <i32: 3> : tile<1xi32>
    %r = reshape %p : tile<ptr<f32>> -> tile<1xptr<f32>>
    %o = offset %r, %off : tile<1xptr<f32>>, tile<1xi32> -> tile<1xptr<f32>>
    %s = reshape %o : tile<1xptr<f32>> -> tile<ptr<f32>>
    %v = constant <f32: 1.0> : tile<f32>
    // expected-error @below {{failed to legalize unresolved target materialization}}
    // expected-note @below {{see existing live user here}}
    %t = store_ptr_tko weak %s, %v : tile<ptr<f32>>, tile<f32> -> token
    return
  }
}
