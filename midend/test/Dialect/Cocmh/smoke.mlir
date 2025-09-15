// RUN: buddy-opt %s | FileCheck %s

// 仅验证方言已注册与 op 可打印（目前只需能 round-trip 解析/打印）。

module {
  // 由于当前 op 需要类型，给出一个最小的 rfft 用例（不会实际执行）：
  // CHECK-LABEL: module
  // CHECK: cocmh.rfft %{{.*}} : memref<?xf64>
  func.func @t() {
    %c4 = arith.constant 4 : index
    %A = memref.alloc(%c4) : memref<?xf64>
    cocmh.rfft %A : memref<?xf64>
    return
  }
}

