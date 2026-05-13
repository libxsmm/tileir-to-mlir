# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "CUDATILE_TO_GPU"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.excludes = ["lit.cfg.py", "lit.site.cfg.py", "CMakeLists.txt"]

config.test_source_root = os.path.join(
    os.path.dirname(__file__), "Conversion", "CudaTileToGPU"
)
config.test_exec_root = os.path.join(config.cudatile_to_gpu_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

tool_dirs = [
    config.cudatile_to_gpu_tool_dir,
    config.llvm_tools_dir,
]

tools = [
    "cudatile-to-gpu",
    "FileCheck",
    "mlir-opt",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
llvm_config.with_environment("PATH", config.cudatile_to_gpu_tool_dir, append_path=True)
