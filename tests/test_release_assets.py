import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "example_workflows" / "xdub_frames_vhs_25fps.json"


class ExampleWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}

    def test_contains_expected_six_node_graph(self):
        self.assertEqual(
            {node["type"] for node in self.nodes.values()},
            {
                "VHS_LoadVideo",
                "LoadAudio",
                "VHS_VideoCombine",
                "XDubFramesLipSync",
                "XDubModelLoader",
                "VAELoader",
            },
        )
        self.assertEqual(len(self.nodes), 6)

        links = {link[0]: link for link in self.workflow["links"]}
        for link_id, (_, source_id, _, target_id, _, _) in links.items():
            self.assertIn(source_id, self.nodes, f"link {link_id} has no source node")
            self.assertIn(target_id, self.nodes, f"link {link_id} has no target node")

        for node in self.nodes.values():
            for node_input in node.get("inputs", []):
                if node_input.get("link") is not None:
                    self.assertIn(node_input["link"], links)
            for output in node.get("outputs", []):
                for link_id in output.get("links") or []:
                    self.assertIn(link_id, links)

    def test_uses_portable_inputs_and_25_fps_audio_trim(self):
        nodes_by_type = {node["type"]: node for node in self.nodes.values()}
        self.assertEqual(nodes_by_type["VHS_LoadVideo"]["widgets_values"]["video"], "video.mp4")
        self.assertEqual(nodes_by_type["LoadAudio"]["widgets_values"][0], "audio.wav")

        combine = nodes_by_type["VHS_VideoCombine"]["widgets_values"]
        self.assertEqual(combine["frame_rate"], 25)
        self.assertTrue(combine["trim_to_audio"])
        self.assertEqual(combine["filename_prefix"], "X-Dub/xdub")
        self.assertNotIn("videopreview", combine)

    def test_has_no_stale_machine_specific_preview(self):
        serialized = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertNotIn("/home/", serialized)
        self.assertNotRegex(serialized, r'"frame_rate"\s*:\s*8(?:\D|$)')


class RegistryReleaseTests(unittest.TestCase):
    def test_pyproject_has_official_comfy_metadata(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertRegex(pyproject, r'(?m)^name\s*=\s*"x-dub"$')
        self.assertRegex(pyproject, r'(?m)^version\s*=\s*"1\.1\.0"$')
        self.assertRegex(pyproject, r'(?m)^license\s*=\s*\{\s*file\s*=\s*"LICENSE"\s*\}$')
        self.assertIn('[tool.comfy]', pyproject)
        self.assertRegex(pyproject, r'(?m)^PublisherId\s*=\s*"darknoah"$')
        self.assertRegex(pyproject, r'(?m)^DisplayName\s*=\s*"X-Dub Lip Sync"$')
        self.assertIn('https://github.com/DarkNoah/comfyui-x-dub', pyproject)

    def test_registry_archive_excludes_local_and_development_files(self):
        patterns = {
            line.strip()
            for line in (ROOT / ".comfyignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue({".venv/", "models/", "tests/", "docs/"}.issubset(patterns))

    def test_publish_action_is_manual_and_uses_official_action(self):
        action = (ROOT / ".github/workflows/publish-comfy-registry.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", action)
        self.assertIn("Comfy-Org/publish-node-action@main", action)
        self.assertIn("secrets.REGISTRY_ACCESS_TOKEN", action)
        self.assertNotRegex(action, r'(?m)^\s+push:\s*$')

    def test_apache_license_is_present(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
