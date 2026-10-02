// RUN: tileir-to-mlir --convert-tileir-to-mlir %s | FileCheck %s

// A ranked pointer tile lowers to its base pointer and the element offsets of
// its lanes from that base.

cuda_tile.module @m {
  // A single-lane pointer tile reshaped to a scalar pointer is its base
  // advanced by the offset of the lane.
  // CHECK-LABEL: gpu.func @reshape_to_scalar(
  // CHECK-SAME:      %[[P:.*]]: memref<*xf32>)
  // CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK:         %[[OFFS:.*]] = vector.broadcast %[[ZERO]] : index to vector<1xindex>
  // CHECK:         %[[SUM:.*]] = arith.addi %[[OFFS]], %{{.*}} : vector<1xindex>
  // CHECK:         %[[LANE:.*]] = vector.extract %[[SUM]][0] : index from vector<1xindex>
  // CHECK:         %[[VIEW:.*]] = memref.reinterpret_cast %[[P]] to offset: [%[[LANE]]], sizes: [1], strides: [1]
  // CHECK:         %[[S:.*]] = memref.cast %[[VIEW]] : {{.*}} to memref<*xf32>
  // CHECK:         memref.reinterpret_cast %[[S]]
  // CHECK:         memref.store
  entry @reshape_to_scalar(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %off = constant <i32: 3> : tile<1xi32>
    %r = reshape %p : tile<ptr<f32>> -> tile<1xptr<f32>>
    %o = offset %r, %off : tile<1xptr<f32>>, tile<1xi32> -> tile<1xptr<f32>>
    %s = reshape %o : tile<1xptr<f32>> -> tile<ptr<f32>>
    %v = constant <f32: 1.0> : tile<f32>
    %t = store_ptr_tko weak %s, %v : tile<ptr<f32>>, tile<f32> -> token
    return
  }

  // A pointer tile advanced in a loop is carried as its base and its offsets.
  // CHECK-LABEL: gpu.func @loop_carried(
  // CHECK-SAME:      %[[BASE:.*]]: memref<*xf32>, %[[OUT:.*]]: memref<*xf32>
  // CHECK:         %[[R:.*]]:3 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[B:.*]] = %[[BASE]], %[[O:.*]] = %{{.*}}, %[[ACC:.*]] = %{{.*}}) -> (memref<*xf32>, vector<16xindex>, vector<16xf32>)
  // CHECK:           %[[M:.*]] = memref.cast %[[B]] : memref<*xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:           %[[V:.*]] = vector.gather %[[M]][%{{.*}}] [%[[O]]]
  // CHECK:           %[[NEXT:.*]] = arith.addi %[[O]], %{{.*}} : vector<16xindex>
  // CHECK:           %[[SUM:.*]] = arith.addf %[[ACC]], %[[V]]
  // CHECK:           scf.yield %[[B]], %[[NEXT]], %[[SUM]] : memref<*xf32>, vector<16xindex>, vector<16xf32>
  // CHECK:         vector.scatter
  entry @loop_carried(%base: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %out: !cuda_tile.tile<!cuda_tile.ptr<f32>>, %n: tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %zero = constant <f32: 0.0> : tile<16xf32>
    %step = constant <i32: 16> : tile<16xi32>
    %iota = iota : tile<16xi32>
    %b1 = reshape %base : tile<ptr<f32>> -> tile<1xptr<f32>>
    %bb = broadcast %b1 : tile<1xptr<f32>> -> tile<16xptr<f32>>
    %p0 = offset %bb, %iota : tile<16xptr<f32>>, tile<16xi32> -> tile<16xptr<f32>>
    %res:2 = for %i in (%c0 to %n, step %c1) : tile<i32>
        iter_values(%p = %p0, %acc = %zero) -> (tile<16xptr<f32>>, tile<16xf32>) {
      %v, %t = load_ptr_tko weak %p : tile<16xptr<f32>> -> tile<16xf32>, !cuda_tile.token
      %next = offset %p, %step : tile<16xptr<f32>>, tile<16xi32> -> tile<16xptr<f32>>
      %sum = addf %acc, %v : tile<16xf32>
      continue %next, %sum : tile<16xptr<f32>>, tile<16xf32>
    }
    %o1 = reshape %out : tile<ptr<f32>> -> tile<1xptr<f32>>
    %ob = broadcast %o1 : tile<1xptr<f32>> -> tile<16xptr<f32>>
    %op = offset %ob, %iota : tile<16xptr<f32>>, tile<16xi32> -> tile<16xptr<f32>>
    %st = store_ptr_tko weak %op, %res#1 : tile<16xptr<f32>>, tile<16xf32> -> !cuda_tile.token
    return
  }

  // An assumption on a pointer tile forwards both its base and its offsets.
  // CHECK-LABEL: gpu.func @assume_ptr_tile(
  // CHECK-SAME:      %[[P:.*]]: memref<*xf32>
  // CHECK:         %[[M:.*]] = memref.cast %[[P]]
  // CHECK:         vector.gather %[[M]]
  entry @assume_ptr_tile(%p: !cuda_tile.tile<!cuda_tile.ptr<f32>>) {
    %iota = iota : tile<8xi32>
    %r = reshape %p : tile<ptr<f32>> -> tile<1xptr<f32>>
    %b = broadcast %r : tile<1xptr<f32>> -> tile<8xptr<f32>>
    %o = offset %b, %iota : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
    %a = assume #cuda_tile.div_by<16>, %o : tile<8xptr<f32>>
    %v, %t = load_ptr_tko weak %a : tile<8xptr<f32>> -> tile<8xf32>, !cuda_tile.token
    %st = store_ptr_tko weak %a, %v : tile<8xptr<f32>>, tile<8xf32> -> !cuda_tile.token
    return
  }
}
