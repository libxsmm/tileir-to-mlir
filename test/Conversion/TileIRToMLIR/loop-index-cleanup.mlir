// RUN: tileir-to-mlir --convert-tileir-to-mlir %s | FileCheck %s
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu' %s | FileCheck %s
// RUN: tileir-to-mlir --convert-tileir-to-mlir --canonicalize-from-tileir %s | FileCheck %s --check-prefix=CANON
// RUN: tileir-to-mlir --convert-tileir-to-mlir='target=cpu' --canonicalize-from-tileir %s | FileCheck %s --check-prefix=CANON

cuda_tile.module @loop_index_cleanup {
  // Dead narrow arithmetic in either branch must not keep the IV cast alive
  // after --canonicalize-from-tileir. Unused reads, stores and region
  // terminators must survive conversion.
  // CHECK-LABEL: {{(gpu|func)\.func}} @conditional_offset
  // CHECK: scf.for %[[IV:.*]] =
  // CHECK-NEXT: arith.index_cast %[[IV]] : index to i32
  // CHECK-NOT: arith.index_cast
  // CHECK: scf.if
  // CHECK-NOT: arith.index_cast
  // CHECK: %[[THEN_OFF:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[THEN_OFF]]]
  // CHECK: memref.load
  // CHECK: memref.store
  // CHECK: } else {
  // CHECK-NOT: arith.index_cast
  // CHECK: %[[ELSE_OFF:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[ELSE_OFF]]]
  // CHECK: memref.store
  // CHECK: }
  // CHECK: }
  // CANON-LABEL: {{(gpu|func)\.func}} @conditional_offset
  // CANON: scf.for %[[IV:.*]] =
  // CANON-NOT: arith.index_cast
  // CANON: scf.if
  // CANON-NOT: arith.index_cast
  // CANON-NOT: arith.muli {{.*}} : i32
  // CANON: %[[THEN_OFF:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[THEN_OFF]]]
  // CANON: memref.store
  // CANON: } else {
  // CANON-NOT: arith.index_cast
  // CANON-NOT: arith.muli {{.*}} : i32
  // CANON: %[[ELSE_OFF:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[ELSE_OFF]]]
  // CANON: memref.store
  // CANON: }
  // CANON: }
  entry @conditional_offset(%p: tile<ptr<i32>>, %n: tile<i32>, %cond: tile<i1>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c4 = constant <i32: 4> : tile<i32>
    for %iv in (%c0 to %n, step %c1) : tile<i32> {
      if %cond {
        %off = muli %iv, %c4 : tile<i32>
        %q = offset %p, %off : tile<ptr<i32>>, tile<i32> -> tile<ptr<i32>>
        %unused, %read_tok = load_ptr_tko weak %q : tile<ptr<i32>> -> tile<i32>, !cuda_tile.token
        %tok = store_ptr_tko weak %q, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
        yield
      } else {
        %off = muli %iv, %c4 : tile<i32>
        %q = offset %p, %off : tile<ptr<i32>>, tile<i32> -> tile<ptr<i32>>
        %tok = store_ptr_tko weak %q, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
        yield
      }
      continue
    }
    return
  }

  // Rebuilding is consumer-local; --canonicalize-from-tileir shares offsets.
  // CHECK-LABEL: {{(gpu|func)\.func}} @shared_offset
  // CHECK: scf.for %[[IV:.*]] =
  // CHECK-NEXT: arith.index_cast %[[IV]] : index to i32
  // CHECK-NOT: arith.index_cast
  // CHECK: %[[OFF0:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[OFF0]]]
  // CHECK-NOT: arith.index_cast
  // CHECK: %[[OFF1:.*]] = arith.muli %[[IV]], %{{.*}} overflow<nsw> : index
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[OFF1]]]
  // CHECK: memref.store
  // CHECK: memref.store
  // CANON-LABEL: {{(gpu|func)\.func}} @shared_offset
  // CANON: %[[FOUR:.*]] = arith.constant 4 : index
  // CANON-NOT: arith.constant 4 : index
  // CANON: scf.for %[[IV:.*]] =
  // CANON-NOT: arith.index_cast
  // CANON-NOT: arith.constant 4 : index
  // CANON: %[[OFF:.*]] = arith.muli %[[IV]], %[[FOUR]] overflow<nsw> : index
  // CANON-NOT: arith.constant 4 : index
  // CANON-NOT: arith.muli
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[OFF]]]
  // CANON-NOT: arith.constant 4 : index
  // CANON-NOT: arith.muli
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[OFF]]]
  // CANON-NOT: arith.muli
  // CANON: memref.store
  // CANON-NOT: arith.muli
  // CANON: memref.store
  // CANON-NOT: arith.muli
  // CANON: }
  entry @shared_offset(%p: tile<ptr<i32>>, %r: tile<ptr<i32>>, %n: tile<i32>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    %c4 = constant <i32: 4> : tile<i32>
    for %iv in (%c0 to %n, step %c1) : tile<i32> {
      %off = muli %iv, %c4 : tile<i32>
      %q = offset %p, %off : tile<ptr<i32>>, tile<i32> -> tile<ptr<i32>>
      %s = offset %r, %off : tile<ptr<i32>>, tile<i32> -> tile<ptr<i32>>
      %tok0 = store_ptr_tko weak %q, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      %tok1 = store_ptr_tko weak %s, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      continue
    }
    return
  }

  // Value-preserving truncation and signed extension support negative offsets.
  // CHECK-LABEL: {{(gpu|func)\.func}} @signed_cast_offset
  // CHECK: %[[LB:.*]] = arith.constant -16 : index
  // CHECK: %[[UB:.*]] = arith.constant 16 : index
  // CHECK: scf.for %[[IV:.*]] = %[[LB]] to %[[UB]]
  // CHECK-NEXT: arith.index_cast %[[IV]] : index to i32
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[IV]]]
  // CHECK: memref.store
  // CHECK: }
  // CANON-LABEL: {{(gpu|func)\.func}} @signed_cast_offset
  // CANON-DAG: %[[LB:.*]] = arith.constant -16 : index
  // CANON-DAG: %[[UB:.*]] = arith.constant 16 : index
  // CANON: scf.for %[[IV:.*]] = %[[LB]] to %[[UB]]
  // CANON-NOT: arith.index_cast
  // CANON-NOT: arith.trunci
  // CANON-NOT: arith.extsi
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[IV]]]
  // CANON: memref.store
  // CANON: }
  entry @signed_cast_offset(%p: tile<ptr<i32>>) {
    %lb = constant <i32: -16> : tile<i32>
    %ub = constant <i32: 16> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    for %iv in (%lb to %ub, step %c1) : tile<i32> {
      %narrow = trunci %iv overflow<none> : tile<i32> -> tile<i8>
      %off = exti %narrow signed : tile<i8> -> tile<i64>
      %q = offset %p, %off : tile<ptr<i32>>, tile<i64> -> tile<ptr<i32>>
      %tok = store_ptr_tko weak %q, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      continue
    }
    return
  }

  // Unsigned extension is value-preserving when the narrow input is non-negative.
  // CHECK-LABEL: {{(gpu|func)\.func}} @unsigned_cast_offset
  // CHECK: scf.for %[[IV:.*]] =
  // CHECK-NEXT: arith.index_cast %[[IV]] : index to i32
  // CHECK: memref.reinterpret_cast %{{.*}} to offset: [%[[IV]]]
  // CHECK: memref.store
  // CHECK: }
  // CANON-LABEL: {{(gpu|func)\.func}} @unsigned_cast_offset
  // CANON: scf.for %[[IV:.*]] =
  // CANON-NOT: arith.index_cast
  // CANON-NOT: arith.trunci
  // CANON-NOT: arith.extui
  // CANON: memref.reinterpret_cast %{{.*}} to offset: [%[[IV]]]
  // CANON: memref.store
  // CANON: }
  entry @unsigned_cast_offset(%p: tile<ptr<i32>>) {
    %c0 = constant <i32: 0> : tile<i32>
    %c100 = constant <i32: 100> : tile<i32>
    %c1 = constant <i32: 1> : tile<i32>
    for %iv in (%c0 to %c100, step %c1) : tile<i32> {
      %narrow = trunci %iv overflow<none> : tile<i32> -> tile<i8>
      %off = exti %narrow unsigned : tile<i8> -> tile<i64>
      %q = offset %p, %off : tile<ptr<i32>>, tile<i64> -> tile<ptr<i32>>
      %tok = store_ptr_tko weak %q, %c1 : tile<ptr<i32>>, tile<i32> -> !cuda_tile.token
      continue
    }
    return
  }
}
