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

// A pointer tile lowers to one base pointer and the offsets of its lanes, so
// selecting lanes of two pointer tiles, which may have different bases, is not
// lowered.

cuda_tile.module @select_ptr_tiles_neg {
  entry @select_ptrs(%a: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %b: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %c = constant <i1: [1, 0, 1, 0]> : tile<4xi1>
    %ra = reshape %a : tile<ptr<f32>> -> tile<1xptr<f32>>
    %pa = broadcast %ra : tile<1xptr<f32>> -> tile<4xptr<f32>>
    %rb = reshape %b : tile<ptr<f32>> -> tile<1xptr<f32>>
    %pb = broadcast %rb : tile<1xptr<f32>> -> tile<4xptr<f32>>
    // expected-error @below {{failed to legalize operation 'cuda_tile.select'}}
    %p = select %c, %pa, %pb : tile<4xi1>, tile<4xptr<f32>>
    %v = constant <f32: 1.0> : tile<4xf32>
    %t = store_ptr_tko weak %p, %v : tile<4xptr<f32>>, tile<4xf32> -> token
    return
  }
}
