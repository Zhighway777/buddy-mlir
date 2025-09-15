# -*- Python -*-

import os
import lit.formats

# Name of this test suite
config.name = "BUDDY-MIDEND"

# Treat .mlir files as shell-style tests
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]

# Source and exec roots
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.buddy_obj_root, "midend", "test")

# Ensure build/bin is in PATH so buddy-opt can be found
bin_dir = os.path.join(config.buddy_obj_root, "bin")
llvm_bin = os.path.join(config.llvm_tools_dir) if hasattr(config, 'llvm_tools_dir') else None
path = bin_dir
if llvm_bin:
    path = path + os.pathsep + llvm_bin
config.environment["PATH"] = path + os.pathsep + config.environment.get("PATH", "")


