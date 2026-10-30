// RUN: tileir-to-mlir --convert-memref-args-to-ptr-args %s | FileCheck %s --check-prefix=FLAT
// RUN: tileir-to-mlir --pack-cpu-kernel-arguments %s | FileCheck %s
// RUN: tileir-to-mlir --pass-pipeline='builtin.module(pack-cpu-kernel-arguments,pack-cpu-kernel-arguments)' %s | FileCheck %s

// FLAT-LABEL: func.func @kernel(
// FLAT-SAME: !llvm.ptr, {{.*}}i1, {{.*}}i8, {{.*}}i16, {{.*}}i32, {{.*}}i64, {{.*}}f32, {{.*}}f64,
// FLAT-NOT: tileir.cpu_arg_count
// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: %[[BUFFER:[^:]+]]: !llvm.ptr,
// CHECK-SAME: %[[X:[^:]+]]: i32,
// CHECK-SAME: %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// CHECK-SAME: tileir.cpu_arg_count = 8 : i64
// CHECK: %[[S0:.*]] = llvm.getelementptr %[[BUFFER]][0]
// CHECK: %[[PTR:.*]] = llvm.load %[[S0]] {{.*}}alignment = 1
// CHECK: llvm.getelementptr %[[BUFFER]][1]
// CHECK: llvm.load {{.*}} -> i1
// CHECK: llvm.getelementptr %[[BUFFER]][2]
// CHECK: llvm.load {{.*}} -> i8
// CHECK: llvm.getelementptr %[[BUFFER]][3]
// CHECK: llvm.load {{.*}} -> i16
// CHECK: llvm.getelementptr %[[BUFFER]][4]
// CHECK: llvm.load {{.*}} -> i32
// CHECK: llvm.getelementptr %[[BUFFER]][5]
// CHECK: llvm.load {{.*}} -> i64
// CHECK: llvm.getelementptr %[[BUFFER]][6]
// CHECK: llvm.load {{.*}} -> f32
// CHECK: llvm.getelementptr %[[BUFFER]][7]
// CHECK: llvm.load {{.*}} -> f64
// CHECK: llvm.store %[[X]], %[[PTR]]
func.func @kernel(%out: !llvm.ptr, %flag: i1, %byte: i8, %short: i16,
                  %integer: i32, %wide: i64, %single: f32, %double: f64,
                  %x: i32, %y: i32, %z: i32, %nx: i32, %ny: i32, %nz: i32) {
  llvm.store %x, %out : i32, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @no_arguments(
// CHECK-SAME: !llvm.ptr,
// CHECK-SAME: tileir.cpu_arg_count = 0 : i64
func.func @no_arguments(%x: i32, %y: i32, %z: i32, %nx: i32, %ny: i32, %nz: i32) {
  return
}

// CHECK-LABEL: func.func private @helper(
// CHECK-SAME: i32)
// CHECK-NOT: tileir.cpu_arg_count
func.func private @helper(%value: i32) {
  return
}

// CHECK: func.func private @external(i32)
func.func private @external(i32)
