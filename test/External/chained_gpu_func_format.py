import os
import shutil

import lit.Test as Test
import lit.TestRunner as TestRunner
import lit.formats

class ChainedGpuFuncCheck(lit.formats.FileBasedTest):
    """Require an upstream RUN line that uses triton-cuda-tile-opt, then run
    a fixed triton-cuda-tile-opt frontend and chain our pipeline + gpu.func check."""

    def __init__(self, our_pipe, check_file):
        super().__init__()
        self.our_pipe = our_pipe
        self.check_file = check_file

    def execute(self, test, litConfig):
        upstream_tool = getattr(test.config, "tileir_upstream_tool_dir", "")
        triton_tool = ""
        if upstream_tool:
            triton_tool = os.path.join(upstream_tool, "triton-cuda-tile-opt")
        if not (triton_tool and os.path.exists(triton_tool)):
            resolved = shutil.which("triton-cuda-tile-opt")
            if not resolved:
                return Test.Result(
                    Test.UNSUPPORTED,
                    "triton-cuda-tile-opt is unavailable; skipping external suite",
                )
            triton_tool = resolved

        parsed = TestRunner.parseIntegratedTestScript(test)
        if isinstance(parsed, Test.Result):
            return parsed

        uses_triton_frontend = False
        for cmd in parsed:
            raw_cmd = getattr(cmd, "command", cmd)
            if not isinstance(raw_cmd, str):
                raw_cmd = str(raw_cmd)
            if "triton-cuda-tile-opt" in raw_cmd:
                uses_triton_frontend = True
                break

        if not uses_triton_frontend:
            return Test.Result(
                Test.UNSUPPORTED,
                "RUN line does not use triton-cuda-tile-opt; skipping external suite",
            )

        # Keep the chaining behavior stable while forcing a fixed frontend.
        script = [
            f"{triton_tool} --split-input-file --convert-triton-to-cuda-tile %s "
            f"| {self.our_pipe} -split-input-file | FileCheck {self.check_file}"
        ]
        if litConfig.noExecute:
            return Test.Result(Test.PASS)

        tmp_dir, tmp_base = TestRunner.getTempPaths(test)
        os.makedirs(os.path.dirname(tmp_base), exist_ok=True)
        subs = TestRunner.getDefaultSubstitutions(test, tmp_dir, tmp_base)
        script = TestRunner.applySubstitutions(
            script, subs, recursion_limit=test.config.recursiveExpansionLimit
        )

        result = TestRunner.executeScript(
            test, litConfig, tmp_base, script, os.path.dirname(tmp_base)
        )
        out = result[0] if len(result) > 0 else ""
        err = result[1] if len(result) > 1 else ""
        exit_code = result[2] if len(result) > 2 else 1
        if exit_code == 0:
            return Test.Result(Test.PASS)
        return Test.Result(Test.FAIL, (out or "") + (err or ""))
