"""Custom-model contracts. All predictions and schema reads are mocked."""
from copy import deepcopy
import os
from pathlib import Path
from unittest.mock import patch
import unittest
import test_provider as original

backend = original.backend
response = original.response


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(CustomModelTests(name) for name in CustomModelTests.__dict__ if name.startswith("test_"))


class CustomModelTests(original.ProviderTests):
    def setUp(self):
        super().setUp()
        self.schemas = {
            "Input": {"type": "object", "required": ["prompt"], "properties": {"prompt": {"type": "string"}}},
            "Output": {"type": "string", "format": "uri"}}
        self.version = "a" * 64
        self.get.side_effect = self.metadata

    def metadata(self, url, **kwargs):
        r = response()
        data = {"id": self.version, "openapi_schema": {"components": {"schemas": deepcopy(self.schemas)}}}
        r.json.return_value = data if "/versions/" in url else {"latest_version": data}
        return r

    def config(self, modality="image", **options):
        return patch("hermes_cli.config.load_config_readonly", return_value={"plugins": {"entries": {
            "replicate": {"settings": {modality + "_config": options}}}}})

    def test_arbitrary_identifiers_all_modalities(self):
        self.assertTrue(self.image.generate("x", model="someone/custom-image")["success"])
        self.assertTrue(self.video.generate("x", model="someone/custom-video")["success"])
        self.schemas["Input"] = {"type": "object", "required": ["audio_file"], "properties": {"audio_file": {"type": "string"}}}
        self.schemas["Output"] = {"type": "object", "properties": {"text": {"type": "string"}}}
        self.post.return_value = response({"text": "custom transcript"})
        self.assertTrue(self.stt.transcribe(str(self.file), model="someone/custom-stt")["success"])
        self.schemas["Input"] = {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}}
        self.schemas["Output"] = {"type": "object", "properties": {"audio": {"type": "string"}}}
        self.post.return_value = response({"audio": "https://replicate.delivery/a.mp3"})
        self.assertTrue(self.tts.synthesize("x", str(Path(self.temp.name)/"out.mp3"), model="someone/custom-tts"))
        self.assertEqual(self.post.call_count, 4)

    def test_custom_video_unknown_duration_is_not_zero_or_wan_default(self):
        # Similar field names alone do not establish an arbitrary model's timing.
        self.schemas["Input"]["properties"].update({
            "num_frames": {"type": "integer", "default": 81},
            "frames_per_second": {"type": "integer", "default": 16}})
        result = self.video.generate("x", model="someone/custom-video")
        self.assertTrue(result["success"], result)
        self.assertIsNone(result["duration"])

    def test_strict_identifiers_no_http(self):
        for model in ("unknown", "https://x/y", "a/b/c", "a/b:abc", "a/b:"+"A"*64, "a/b?x=1", "a/b\n", "../foo"):
            self.assert_failure(lambda: self.image.generate("x", model=model), "invalid_input")
        self.post.assert_not_called()
        self.get.assert_not_called()

    def test_cache_profile_ttl_and_bound(self):
        backend.discover("someone/model")
        backend.discover("someone/model")
        self.assertEqual(self.get.call_count, 1)
        with patch.dict(os.environ, {"HERMES_HOME": str(Path(self.temp.name)/"other")}):
            backend.discover("someone/model")
        self.assertEqual(self.get.call_count, 2)
        with patch.object(backend, "_SCHEMA_TTL", -1):
            backend.discover("someone/model")
        self.assertEqual(self.get.call_count, 3)
        with patch.object(backend, "_SCHEMA_CACHE_SIZE", 2):
            for n in range(4):
                backend.discover(f"someone/model-{n}")
        self.assertEqual(len(backend._SCHEMA_CACHE), 2)

    def test_pinned_versions_and_mismatch(self):
        with self.config(version=self.version):
            self.assertTrue(self.image.generate("x", model="someone/model")["success"])
        self.assertTrue(self.get.call_args.args[0].endswith("/versions/"+self.version))
        self.assertEqual(self.post.call_args.kwargs["json"]["version"], self.version)
        self.post.reset_mock()
        with self.config(version="b"*64):
            self.assert_failure(lambda: self.image.generate("x", model="someone/model:"+self.version), "invalid_input")
        self.assert_failure(lambda: self.image.generate("x", model="someone/model:"+"c"*64), "schema_error")
        self.post.assert_not_called()

    def test_required_enum_type_range_unknown_and_secrets_zero_post(self):
        self.schemas["Input"]["required"].append("choice")
        self.schemas["Input"]["properties"].update({"choice": {"type": "string", "enum": ["yes"]},
            "count": {"type": "integer", "minimum": 1, "maximum": 3}, "flag": {"type": "boolean"}})
        for extras in ({}, {"choice":"no"}, {"choice":"yes", "count": True},
                       {"choice":"yes", "count":4}, {"choice":"yes", "flag":"true"},
                       {"choice":"yes", "unknown":1}, {"api_key":"bad"}):
            with self.config(extra_inputs=extras):
                self.assert_failure(lambda: self.image.generate("x", model="someone/model"), "invalid_input")
        self.post.assert_not_called()
        with self.config(extra_inputs={"choice":"yes", "count":2, "flag":True}):
            self.assertTrue(self.image.generate("x", model="someone/model")["success"])

    def test_nonstandard_mapping_and_output_path(self):
        self.schemas["Input"] = {"type":"object", "required":["description"], "properties":{"description":{"type":"string"}}}
        self.schemas["Output"] = {"type":"object", "properties":{"result":{"type":"array", "items":{"type":"string"}}}}
        self.post.return_value = response({"result":["https://replicate.delivery/a.png"]})
        with self.config(input_mapping={"prompt":"description"}, output_mapping="result.0"):
            self.assertTrue(self.image.generate("x", model="someone/model")["success"])
        self.assertEqual(self.post.call_args.kwargs["json"]["input"], {"description":"x"})
        self.post.reset_mock()
        with self.config(input_mapping={"prompt":"description"}, output_mapping="missing"):
            self.assert_failure(lambda: self.image.generate("x", model="someone/model"), "invalid_input")
        self.post.assert_not_called()

    def test_ambiguous_input_and_output_preflight(self):
        self.schemas["Input"] = {"type":"object", "properties":{"text":{"type":"string"}, "input_text":{"type":"string"}}}
        self.assert_failure(lambda: self.tts.synthesize("x", "x", model="someone/tts"), "invalid_input")
        self.schemas["Output"] = {"type":"object", "properties":{"image":{"type":"string"}, "url":{"type":"string"}}}
        with self.config(modality="tts", input_mapping={"text":"text"}):
            self.assert_failure(lambda: self.tts.synthesize("x", "x", model="someone/other"), "invalid_input")
        self.post.assert_not_called()

    def test_discovery_failures_and_external_refs(self):
        self.get.return_value = response()
        self.get.side_effect = None
        self.get.return_value.status_code = 404
        import requests
        self.get.return_value.raise_for_status.side_effect = requests.HTTPError(response=self.get.return_value)
        self.assert_failure(lambda: self.image.generate("x", model="someone/missing"), "schema_error")
        self.get.side_effect = self.metadata
        self.schemas["Input"]["properties"]["prompt"] = {"$ref":"https://evil.test/schema"}
        self.assert_failure(lambda: self.image.generate("x", model="someone/external"), "schema_error")
        self.post.assert_not_called()

    def test_required_fields_fail_all_modalities_before_post(self):
        self.schemas["Input"] = {"type":"object", "required":["special"], "properties":{
            "prompt":{"type":"string"}, "text":{"type":"string"}, "audio":{"type":"string"},
            "special":{"type":"integer"}}}
        calls = (lambda:self.image.generate("x",model="someone/image"),
                 lambda:self.video.generate("x",model="someone/video"),
                 lambda:self.stt.transcribe(str(self.file),model="someone/stt"),
                 lambda:self.tts.synthesize("x","x",model="someone/tts"))
        for call in calls:
            self.assert_failure(call,"invalid_input")
        self.post.assert_not_called()

    def test_cached_missing_credentials_all_modalities_no_post(self):
        backend.discover("someone/model")
        self.secret.return_value = None
        for call in (lambda:self.image.generate("x",model="someone/model"),
                     lambda:self.video.generate("x",model="someone/model"),
                     lambda:self.stt.transcribe(str(self.file),model="someone/model"),
                     lambda:self.tts.synthesize("x","x",model="someone/model")):
            self.assert_failure(call,"missing_api_key")
        self.post.assert_not_called()
