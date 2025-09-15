//===- CocmhOps.td - cocmh Dialect Ops -------------------------*- tablegen -*-===//
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
// This is the top level file for operations in the Cocmh dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COCMH_COCMHOPS_TD
#define COCMH_COCMHOPS_TD

// CocmhOps.td
include "Cocmh/CocmhDialect.td"
include "mlir/IR/OpBase.td"

def Cocmh_RFFTOp : Cocmh_Op<"rfft"> {
  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref);
  let assemblyFormat = [{
    $memref attr-dict `:` type($memref)
  }];
}

#endif // COCMH_COCMHOPS_TD
