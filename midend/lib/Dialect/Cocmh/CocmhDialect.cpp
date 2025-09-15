//===-- CocmhDialect.cpp - cocmh Dialect Implementation ---------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the core dialect structure for the Compiler Multi-core Co-design IR
// (Cocmh), including dialect registration and initialization.
//
//===----------------------------------------------------------------------===//

// 头部包含顺序：先 MLIR/项目头，再生成的 inc
#include "mlir/IR/DialectImplementation.h"
#include "Cocmh/CocmhDialect.h"
#include "Cocmh/CocmhOps.h"
#include "Cocmh/CocmhDialect.cpp.inc"

using namespace mlir;
using namespace buddy::cocmh;

void CocmhDialect::initialize() {
  addOperations<
    // 自动展开生成
  #define GET_OP_LIST
  #include "Cocmh/CocmhOps.cpp.inc"
  >();
  // 如有类型/属性：addTypes<...>(); addAttributes<...>();
}