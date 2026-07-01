# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "TILEIR_TO_MLIR"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

config.test_source_root = os.path.join(
    os.path.dirname(__file__), "Conversion", "TileIRToMLIR"
)
config.test_exec_root = os.path.join(config.tileir_to_mlir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

tool_dirs = [
    config.tileir_to_mlir_tool_dir,
    config.llvm_tools_dir,
]

tools = [
    "tileir-to-mlir",
    "FileCheck",
    "mlir-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
llvm_config.with_environment("PATH", config.tileir_to_mlir_tool_dir, append_path=True)
