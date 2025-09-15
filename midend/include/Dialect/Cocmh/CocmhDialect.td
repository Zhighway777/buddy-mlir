//===- CocmhDialect.td - cocmh Dialect Definition --------------*- tablegen -*-===//
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
// This is the top level file for the Cocmh dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COCMH_COCMHDIALECT_TD
#define COCMH_COCMHDIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Cocmh Dialect Definition.
//===----------------------------------------------------------------------===//

def Cocmh_Dialect : Dialect {
  let name = "cocmh";
  let summary = "The Cocmh Dialect.";
  let description = [{
    Compiler Multi-core Hardware Co-design dialect(`Cocmh`) dialect is created for the purpose of developing a MLIR backend for performing hardware co-design operations.
  }];
  let cppNamespace = "::buddy::cocmh";
}

//===----------------------------------------------------------------------===//
// Base Cocmh Operation Definition.
//===----------------------------------------------------------------------===//
//TODO: need to complete the Cocmh operations
class Cocmh_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cocmh_Dialect, mnemonic, traits>;

#endif // COCMH_COCMHDIALECT_TD
