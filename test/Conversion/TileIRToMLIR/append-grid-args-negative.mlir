// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu append-grid-args=false' --verify-diagnostics %s

// On non-GPU targets, dim-query ops require append-grid-args=true.

cuda_tile.module @m {
  entry @coords(%arg0: tile<i32>) {
    // expected-error @below {{failed to legalize operation 'cuda_tile.get_tile_block_id'}}
    %bx, %by, %bz = get_tile_block_id : tile<i32>
    %nx, %ny, %nz = get_num_tile_blocks : tile<i32>
    %s0 = addi %bx, %nx : tile<i32>
    %s1 = addi %by, %nz : tile<i32>
    %s2 = addi %s0, %arg0 : tile<i32>
    return
  }
}
