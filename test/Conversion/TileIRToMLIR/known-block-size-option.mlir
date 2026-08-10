// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu known-block-size=128,2,1' %s | FileCheck %s --check-prefix=SET
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=gpu' %s | FileCheck %s --check-prefix=UNSET
// RUN: not tileir-to-mlir --convert-tileir-to-mlir='target=gpu known-block-size=128,2' %s 2>&1 | FileCheck %s --check-prefix=ERR

// Verifies the `known-block-size` option: when three values are given they are
// attached to every generated gpu.func as the `known_block_size` attribute;
// when the option is omitted the attribute is not set; any other number of
// values is rejected.

// SET: gpu.func @add(%{{.*}}: i32, %{{.*}}: i32) kernel attributes {known_block_size = array<i32: 128, 2, 1>}

// UNSET-NOT: known_block_size

// ERR: 'known-block-size' expects exactly three values

cuda_tile.module @m {
  entry @add(%a: !cuda_tile.tile<i32>, %b: !cuda_tile.tile<i32>) {
    %r = addi %a, %b : tile<i32>
    return
  }
}
