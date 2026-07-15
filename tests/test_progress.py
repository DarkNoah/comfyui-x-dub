import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RecordingProgressBar:
    instances = []

    def __init__(self, total):
        self.total = total
        self.values = []
        self.__class__.instances.append(self)

    def update_absolute(self, value):
        self.values.append(value)


def load_nodes():
    RecordingProgressBar.instances.clear()

    numpy = types.ModuleType("numpy")
    torch = types.ModuleType("torch")
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = lambda *args, **kwargs: None
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.save_file = lambda *args, **kwargs: None
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda category: []
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = RecordingProgressBar

    stubs = {
        "numpy": numpy,
        "torch": torch,
        "safetensors": safetensors,
        "safetensors.torch": safetensors_torch,
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.utils": comfy_utils,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("xdub_nodes_progress_test", ROOT / "nodes.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def load_runner():
    spec = importlib.util.spec_from_file_location("xdub_runner_progress_test", ROOT / "xdub_runtime" / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParentProgressTests(unittest.TestCase):
    def test_forwards_structured_progress_to_comfyui(self):
        module = load_nodes()
        tracker = module._XDubProgressTracker()

        tracker.consume("[X-Dub Progress] 15\n")
        tracker.consume("[X-Dub Progress] 85\n")
        tracker.complete()

        self.assertEqual(len(RecordingProgressBar.instances), 1)
        self.assertEqual(RecordingProgressBar.instances[0].total, 100)
        self.assertEqual(RecordingProgressBar.instances[0].values, [0, 15, 85, 100])

    def test_ignores_malformed_out_of_range_and_decreasing_progress(self):
        module = load_nodes()
        tracker = module._XDubProgressTracker()

        tracker.consume("Denoising step 1/30\n")
        tracker.consume("[X-Dub Progress] 101\n")
        tracker.consume("[X-Dub Progress] -1\n")
        tracker.consume("[X-Dub Progress] 20\n")
        tracker.consume("[X-Dub Progress] 10\n")
        tracker.consume("[X-Dub Progress] 20.5\n")

        self.assertEqual(RecordingProgressBar.instances[0].values, [0, 20])


class RunnerProgressTests(unittest.TestCase):
    def test_report_progress_emits_machine_readable_protocol(self):
        runner = load_runner()
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            runner.report_progress(15)

        self.assertEqual(output.getvalue(), "[X-Dub Progress] 15\n")

    def test_diffusion_progress_accumulates_across_clips(self):
        runner = load_runner()
        values = []
        runner.report_progress = values.append

        with contextlib.redirect_stdout(io.StringIO()):
            first_clip = list(runner.console_progress(["a", "b"], clip_index=0, num_clips=2))
            second_clip = list(runner.console_progress(["c", "d"], clip_index=1, num_clips=2))

        self.assertEqual(first_clip, ["a", "b"])
        self.assertEqual(second_clip, ["c", "d"])

        self.assertEqual(values, [40, 55, 70, 85])


if __name__ == "__main__":
    unittest.main(verbosity=2)
