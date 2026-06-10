import os
import sys

import lit.formats
from lit.llvm import llvm_config

sys.path.insert(0, os.path.dirname(__file__))
from chained_gpu_func_format import ChainedGpuFuncCheck

config.name = "CUDATILE_TO_GPU_EXTERNAL"
config.suffixes = [".mlir"]
config.test_source_root = config.tileir_external_test_dir
config.test_exec_root = os.path.join(
    config.cudatile_to_gpu_obj_root, "test", "External"
)

# Our pipeline, chained onto each file's own front-end command.
OUR_PIPE = "cudatile-to-gpu --tileir-ptr-to-view --convert-cuda-tile-to-gpu"
CHECK_FILE = os.path.join(os.path.dirname(__file__), "gpu-func.check")

# Files whose RUN lines intentionally emit diagnostics / no gpu.func.
config.excludes = ["op-conversion-xfailure.mlir", 
                "fma.mlir",
                "inliner.mlir",
                "op-conversion-assume.mlir",
                "op-conversion-auto-memtoken.mlir",
                "op-conversion-modifiers.mlir",
                "op-conversion.mlir",
                "op-rewrite-assume.mlir",
                "op-conversion-barrier.mlir",]

config.test_format = ChainedGpuFuncCheck(OUR_PIPE, CHECK_FILE)

tools = ["triton-cuda-tile-opt", "cudatile-to-gpu", "FileCheck", "mlir-opt"]
llvm_config.add_tool_substitutions(
    tools,
    [
        config.tileir_upstream_tool_dir,
        config.cudatile_to_gpu_tool_dir,
        config.llvm_tools_dir,
    ],
)
