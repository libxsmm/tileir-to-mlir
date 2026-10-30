// RUN: tileir-to-mlir --pack-cpu-kernel-arguments --split-input-file --verify-diagnostics %s

// expected-error @+1 {{expected a void CPU kernel with six trailing i32 grid arguments}}
func.func @missing_grid(%value: i32) {
  return
}

// -----

// expected-error @+1 {{CPU packed arguments must be host pointers}}
func.func @unpromoted(%arg: memref<*xf32>,
                     %x: i32, %y: i32, %z: i32, %nx: i32, %ny: i32, %nz: i32) {
  return
}

// -----

// expected-error @+1 {{cannot pack a CPU kernel with in-module symbol uses}}
func.func @referenced(%x: i32, %y: i32, %z: i32, %nx: i32, %ny: i32, %nz: i32) {
  return
}
func.func private @caller(%x: i32) {
  func.call @referenced(%x, %x, %x, %x, %x, %x) : (i32, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

// expected-error @+1 {{expected a void CPU kernel with six trailing i32 grid arguments}}
func.func @result(%x: i32, %y: i32, %z: i32, %nx: i32, %ny: i32, %nz: i32) -> i32 {
  return %x : i32
}
