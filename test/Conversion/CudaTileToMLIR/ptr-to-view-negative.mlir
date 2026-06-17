// RUN: cudatile-to-mlir --tileir-ptr-to-view --verify-diagnostics %s | FileCheck %s

module {
  cuda_tile.module @cuda_tile_module {

    // CHECK-LABEL: entry @no_mask
    entry @no_mask(%arg0: tile<ptr<f32>>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[NM:.*]], %[[NMT:.*]] = load_ptr_tko weak %[[NMP:.*]] : tile<1024xptr<f32>> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: load has no mask; skipping}}
      %tile, %token = load_ptr_tko weak %ptr : tile<1024xptr<f32>> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @bad_padding
    entry @bad_padding(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %pad = constant <f32: 1.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[BP:.*]], %[[BPT:.*]] = load_ptr_tko weak %[[BPP:.*]], %[[BPM:.*]], %{{.*}} : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: padding value not recognised}}
      %tile, %token = load_ptr_tko weak %ptr, %mask, %pad : tile<1024xptr<f32>>, tile<1024xi1>, tile<1024xf32> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @non_canonical_addend
    entry @non_canonical_addend(%arg0: tile<ptr<f32>>, %arg1: tile<i32>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %cst_5 = constant <i32: 5> : tile<1024xi32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %shape_1d = reshape %arg1 : tile<i32> -> tile<1xi32>
      %shape_bc = broadcast %shape_1d : tile<1xi32> -> tile<1024xi32>
      %mask = cmpi less_than %index, %shape_bc, signed : tile<1024xi32> -> tile<1024xi1>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %cst_5 : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[NC:.*]], %[[NCT:.*]] = load_ptr_tko weak %[[NCP:.*]], %[[NCM:.*]] : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, token
      // expected-remark @below {{tileir-ptr-to-view: pointer-arithmetic pattern not recognised; skipping}}
      %tile, %token = load_ptr_tko weak %ptr, %mask : tile<1024xptr<f32>>, tile<1024xi1> -> tile<1024xf32>, !cuda_tile.token
      return
    }

    // CHECK-LABEL: entry @store_no_mask
    entry @store_no_mask(%arg0: tile<ptr<f32>>) {
      %cst_1024 = constant <i32: 1024> : tile<i32>
      %value = constant <f32: 1.000000e+00> : tile<1024xf32>
      %block_id_x, %block_id_y, %block_id_z = get_tile_block_id : tile<i32>
      %start = muli %block_id_x, %cst_1024 : tile<i32>
      %lane = iota : tile<1024xi32>
      %start_1d = reshape %start : tile<i32> -> tile<1xi32>
      %start_bc = broadcast %start_1d : tile<1xi32> -> tile<1024xi32>
      %index = addi %start_bc, %lane : tile<1024xi32>
      %base_1d = reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base_bc = broadcast %base_1d : tile<1xptr<f32>> -> tile<1024xptr<f32>>
      %ptr = offset %base_bc, %index : tile<1024xptr<f32>>, tile<1024xi32> -> tile<1024xptr<f32>>
      // CHECK: %[[SNM:.*]] = store_ptr_tko weak %[[SNMP:.*]], %{{.*}} : tile<1024xptr<f32>>, tile<1024xf32> -> token
      // expected-remark @below {{tileir-ptr-to-view: store has no mask; skipping}}
      %token = store_ptr_tko weak %ptr, %value : tile<1024xptr<f32>>, tile<1024xf32> -> !cuda_tile.token
      return
    }
  }
}