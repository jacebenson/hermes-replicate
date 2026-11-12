"""Offline contracts using real installed Hermes APIs, never replacement modules."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from shutil import copyfile as shutil_copyfile
from unittest.mock import MagicMock, patch
from typing import Any
import requests

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replicate_backend", ROOT / "__init__.py")
assert spec is not None and spec.loader is not None
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
REAL_SAVE_AUDIO = backend.save_audio


def response(output: Any="https://replicate.delivery/test.png", status="succeeded"):
    result = MagicMock()
    result.status_code = 200
    result.json.return_value = {"id": "test-id", "status": status, "output": output}
    return result


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.secret = patch.object(backend, "get_secret", return_value="offline-token").start()
        patch("hermes_cli.config.load_config_readonly", return_value={}).start()
        self.post = patch.object(backend.requests, "post", return_value=response()).start()
        def metadata_get(url, **kwargs):
            if "/models/" in url:
                model = url.split("/models/", 1)[1].split("/versions/", 1)[0]
                item = backend.CATALOG[model]
                data = {"id": item["version"], "openapi_schema": {"components": {"schemas": item["schemas"]}}}
                result = response()
                result.json.return_value = data if "/versions/" in url else {"latest_version": data}
                return result
            return response()
        self.get = patch.object(backend.requests, "get", side_effect=metadata_get).start()
        backend._SCHEMA_CACHE.clear()
        self.save = patch.object(backend, "save_url_image", return_value=Path("/offline/image.png")).start()
        patch.object(backend, "save_url_video", return_value=Path("/offline/video.mp4")).start()
        self.audio = patch.object(backend, "save_audio", return_value=Path("/offline/audio.mp3")).start()
        patch.object(backend.shutil, "copyfile").start()
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.file = Path(self.temp.name) / "audio.wav"
        self.file.write_bytes(b"offline-audio")
        self.home = patch.dict(os.environ, {"HERMES_HOME": self.temp.name}).start()
        self.addCleanup(patch.stopall)
        self.image = backend.ReplicateImageGenProvider()
        self.video = backend.ReplicateVideoGenProvider()
        self.stt = backend.ReplicateTranscriptionProvider()
        self.tts = backend.ReplicateTTSProvider()

    def calls(self):
        return [lambda: self.image.generate("test"), lambda: self.video.generate("test"),
                lambda: self.stt.transcribe(str(self.file)),
                lambda: self.tts.synthesize("test", str(Path(self.temp.name) / "out.mp3"))]

    def assert_failure(self, call, kind):
        try:
            result = call()
        except backend.ReplicateError as exc:
            self.assertEqual(exc.kind, kind)
            self.assertNotIn("sensitive", str(exc))
        else:
            self.assertFalse(result["success"], result)
            self.assertEqual(result["error_type"], kind, result)
            self.assertNotIn("sensitive", result["error"])

    def test_native_registration_and_actual_abc(self):
        from hermes_cli.plugins_context import PluginContext
        ctx = MagicMock(spec=PluginContext)
        backend.register(ctx)
        for name, abc in [("image_gen", backend.ImageGenProvider), ("video_gen", backend.VideoGenProvider),
                          ("transcription", backend.TranscriptionProvider), ("tts", backend.TTSProvider)]:
            self.assertIsInstance(getattr(ctx, f"register_{name}_provider").call_args.args[0], abc)
        self.assertEqual(len(self.image.list_models()), 12)
        self.assertEqual(len(set(backend.IMAGE_MODELS)), 12)
        self.assertFalse(self.stt.streaming_capable)
        self.assertFalse(self.tts.streams_pcm)

    def test_real_settings_renderer_masks_token_and_lists_models(self):
        from hermes_cli.plugins_settings import plugin_settings_fields
        with patch("hermes_cli.config.get_env_value", return_value="never-render-this-secret"):
            fields = plugin_settings_fields("replicate", ROOT)
        token = next(f for f in fields if f["key"] == "api_token")
        self.assertEqual(token["type"], "secret")
        self.assertEqual(token["env"], "REPLICATE_API_TOKEN")
        self.assertTrue(token["has_value"])
        self.assertNotIn("value", token)
        self.assertNotIn("never-render-this-secret", json.dumps(fields))
        image = next(f for f in fields if f["key"] == "image_model")
        self.assertEqual(image["type"], "string")
        self.assertFalse(image.get("choices"))

    def test_every_image_model_payload_output_and_schema(self):
        for model in backend.IMAGE_MODELS:
            schema = backend.CATALOG[model]["output"]
            output = ["https://replicate.delivery/image.png"] if schema["type"] == "array" else "https://replicate.delivery/image.png"
            self.post.return_value = response(output)
            for aspect in ("landscape", "square", "portrait"):
                with self.subTest(model=model, aspect=aspect):
                    result = self.image.generate(" illustration ", aspect, model=model)
                    self.assertTrue(result["success"], result)
                    payload = self.post.call_args.kwargs["json"]
                    self.assertEqual(payload["version"], backend.CATALOG[model]["version"])
                    self.assertEqual(payload["input"]["prompt"], "illustration")
                    if model == backend.ANIMAGINE:
                        self.assertEqual((payload["input"]["width"], payload["input"]["height"]), backend._SIZES[aspect])
                    else:
                        self.assertEqual(payload["input"]["aspect_ratio"], backend._RATIOS[aspect])
                    backend.validate_input(model, payload["input"])
                    self.assertEqual(result["image"], "/offline/image.png")

    def test_video_mapping(self):
        result = self.video.generate("motion", resolution="720p", seed=12)
        self.assertTrue(result["success"], result)
        payload = self.post.call_args.kwargs["json"]
        self.assertEqual(payload["version"], backend.CATALOG[backend.VIDEO_MODEL]["version"])
        self.assertEqual(payload["input"], {"prompt": "motion", "aspect_ratio": "16:9", "resolution": "720p",
                                          "duration": 5, "draft": True, "fps": 24,
                                          "prompt_upsampling": False, "seed": 12})
        self.assertEqual(result["video"], "/offline/video.mp4")
        self.assertEqual(result["duration"], 5)

    def test_video_duration_uses_effective_configured_duration(self):
        config = {"video_gen": {"replicate": {"extra_inputs": {
            "duration": 4, "draft": False, "fps": 48}}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            result = self.video.generate("motion")
        self.assertTrue(result["success"], result)
        self.assertEqual(result["duration"], 4.0)

    def test_stt_mapping_and_durable_transcript(self):
        self.post.return_value = response({"transcription": "Hello world", "detected_language": "en"})
        result = self.stt.transcribe(str(self.file), language="en", prompt="vocabulary")
        self.assertTrue(result["success"], result)
        self.assertEqual(Path(result["path"]).read_text(), "Hello world")
        data = self.post.call_args.kwargs["json"]["input"]
        self.assertTrue(data["audio"].startswith("data:audio/"))
        self.assertEqual(data["initial_prompt"], "vocabulary")
        self.assertEqual(data["language"], "en")
        self.assertEqual(data["transcription"], "plain text")

    def test_tts_mapping_and_formats(self):
        for format in ("mp3", "wav", "flac"):
            result = self.tts.synthesize(" Hello ", str(Path(self.temp.name) / "out.mp3"), format=format, speed=1.2)
            self.assertTrue(result.endswith("." + format))
            self.assertEqual(self.post.call_args.kwargs["json"]["input"],
                {"text": "Hello", "voice_id": "English_Wiselady", "speed": 1.2, "audio_format": format})

    def test_invalid_inputs_all_modalities_before_http(self):
        calls = [lambda: self.image.generate(" "), lambda: self.image.generate("a", model="unknown"),
                 lambda: self.image.generate("a", image_url="x"), lambda: self.image.generate("a", options={"steps": -1}),
                 lambda: self.video.generate(""), lambda: self.video.generate("a", model="unknown"),
                 lambda: self.video.generate("a", duration=999), lambda: self.video.generate("a", audio=True),
                 lambda: self.video.generate("a", aspect_ratio="invalid"), lambda: self.video.generate("a", resolution="invalid"),
                 lambda: self.video.generate("a", seed="bad"), lambda: self.stt.transcribe("/missing.wav"),
                 lambda: self.stt.transcribe(str(self.file), language="en-US"),
                 lambda: self.stt.transcribe(str(self.file), model="unknown"),
                 lambda: self.tts.synthesize("", "x"), lambda: self.tts.synthesize("a", "x", speed=9),
                 lambda: self.tts.synthesize("a", "x", format="opus"), lambda: self.tts.synthesize("a"*10001, "x"),
                 lambda: self.tts.synthesize("a", "x", model="unknown")]
        for call in calls:
            self.assert_failure(call, "invalid_input")
        self.post.assert_not_called()

    def test_missing_and_blank_profile_credentials_all_modalities(self):
        for token in (None, " "):
            self.secret.return_value = token
            self.assertFalse(self.image.is_available())
            for call in self.calls():
                self.assert_failure(call, "missing_api_key")
        self.post.assert_not_called()

    def test_real_secret_scope_beats_environment(self):
        from agent import secret_scope
        token = secret_scope.set_secret_scope({"REPLICATE_API_TOKEN": "scoped-token"})
        try:
            with patch.object(backend, "get_secret", wraps=secret_scope.get_secret), patch.dict(os.environ, {"REPLICATE_API_TOKEN": "wrong-token"}):
                self.assertTrue(self.image.generate("test")["success"])
                self.assertEqual(self.post.call_args.kwargs["headers"]["Authorization"], "Bearer scoped-token")
        finally:
            secret_scope.reset_secret_scope(token)

    def test_configured_model_never_falls_back(self):
        config = {"plugins": {"entries": {"replicate": {"settings": {"image_model": backend.ANIMAGINE}}}}}
        with patch("hermes_cli.config.load_config_readonly", return_value=config):
            self.assertEqual(self.image.generate("test")["model"], backend.ANIMAGINE)
        with patch("hermes_cli.config.load_config_readonly", return_value={"image_gen": {"replicate": {"model": "unknown"}}}):
            self.assert_failure(lambda: self.image.generate("test"), "invalid_input")

    def test_sanitized_errors_all_modalities(self):
        http = requests.Response(); http.status_code = 401
        for error, kind in [(requests.HTTPError("sensitive", response=http), "api_error"),
                            (requests.Timeout("sensitive"), "timeout"),
                            (requests.ConnectionError("sensitive"), "connection_error")]:
            self.post.side_effect = error
            for call in self.calls():
                self.assert_failure(call, kind)

    def test_poll_and_deadline_all_modalities(self):
        for call in self.calls():
            self.post.return_value = response(status="starting")
            with patch.object(backend.time, "monotonic", side_effect=[0, 0, 301]):
                self.assert_failure(call, "timeout")
        self.post.return_value = response(status="processing")
        with patch.object(backend.time, "sleep"):
            self.assertTrue(self.image.generate("test")["success"])
        self.assertEqual(self.get.call_args.args[0], backend._API + "/test-id")
        self.assertFalse(self.get.call_args.kwargs["allow_redirects"])

    def test_failed_canceled_and_malformed_all_modalities(self):
        for state in ("failed", "canceled", "unexpected"):
            self.post.return_value = response(status=state)
            for call in self.calls():
                self.assert_failure(call, "invalid_response" if state == "unexpected" else "api_error")
        for data in ([], {}, {"status": "succeeded"}):
            self.post.return_value.json.return_value = data
            for call in self.calls():
                self.assert_failure(call, "invalid_response")

    def test_bad_media_and_cache_errors(self):
        for output in ([None], "http://example.com/a", {"url": "x"}):
            self.post.return_value = response(output)
            for call in (self.calls()[0], self.calls()[1], self.calls()[3]):
                self.assert_failure(call, "invalid_response")
        self.post.return_value = response([])
        self.assert_failure(self.calls()[0], "empty_response")
        self.post.return_value = response()
        self.save.side_effect = OSError("sensitive")
        self.assert_failure(self.calls()[0], "cache_error")

    def test_remaining_cache_failures_and_stt_output_validation(self):
        with patch.object(backend, "save_url_video", side_effect=OSError("sensitive")):
            self.assert_failure(self.calls()[1], "cache_error")
        self.audio.side_effect = OSError("sensitive")
        self.assert_failure(self.calls()[3], "cache_error")
        for output in (None, {}, {"transcription": None}):
            self.post.return_value = response(output)
            self.assert_failure(self.calls()[2], "invalid_response")
        self.post.return_value = response({"transcription": ""})
        self.assertTrue(self.calls()[2]()["success"])

    def test_real_cache_helpers_mocked_http(self):
        from agent.image_gen_provider import save_url_image
        from agent.video_gen_provider import save_url_video
        client = MagicMock()
        download = client.stream.return_value.__enter__.return_value
        download.status_code = 200; download.is_success = True
        for saver, mime, folder in [(save_url_image, "image/png", "images"),
                                    (save_url_video, "video/mp4", "videos")]:
            download.headers = {"Content-Type": mime}
            download.iter_bytes.return_value = iter([b"offline-bytes"])
            with patch("tools.url_safety.is_safe_url", return_value=True), patch("tools.url_safety.create_ssrf_safe_client", return_value=client):
                path = saver("https://replicate.delivery/test")
            self.assertEqual(path.read_bytes(), b"offline-bytes")
            self.assertTrue(path.is_relative_to(Path(self.temp.name) / "cache/generated" / folder))
            self.assertIsNone(client.stream.call_args.kwargs["headers"])

    def audio_download(self, chunks=(b"speech",), *, status=200, headers=None, error=None):
        client = MagicMock()
        client.__aenter__.return_value = client
        download = client.stream.return_value.__aenter__.return_value
        download.status_code = status
        download.headers = headers if headers is not None else {"Content-Type": "audio/mpeg"}

        async def iterate(**kwargs):
            self.assertEqual(kwargs, {"chunk_size": 65536})
            for chunk in chunks:
                yield chunk
            if error:
                raise error
        download.aiter_bytes = iterate
        return client

    def test_audio_public_transport_and_profile_cache(self):
        for mime, extension in backend._AUDIO_TYPES.items():
            client = self.audio_download(headers={"Content-Type": mime + "; charset=binary"})
            with patch.object(backend, "is_safe_url", return_value=True), patch.object(
                    backend, "create_ssrf_safe_async_client", return_value=client) as factory:
                path = REAL_SAVE_AUDIO("https://replicate.delivery/test")
            self.assertEqual(path.read_bytes(), b"speech")
            self.assertEqual(path.suffix, "." + extension)
            self.assertTrue(path.is_relative_to(Path(self.temp.name) / "cache/generated/audio"))
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            factory.assert_called_once_with(timeout=60, follow_redirects=False, trust_env=False)
            client.stream.assert_called_once_with("GET", "https://replicate.delivery/test", headers=None)

    def test_audio_rejects_redirects_types_empty_oversize_and_cleans_partial(self):
        cases = [dict(status=302, headers={"Location": "http://127.0.0.1/private"}),
                 dict(status=401), dict(headers={"Content-Type": "text/html"}),
                 dict(headers={}), dict(chunks=()), dict(chunks=(b"12345",)),
                 dict(headers={"Content-Type": "audio/mpeg", "Content-Length": "5"}),
                 dict(headers={"Content-Type": "audio/mpeg", "Content-Length": "bad"}),
                 dict(chunks=(b"ok",), error=OSError("interrupted")),
                 dict(chunks=(b"ok",), error=backend.asyncio.CancelledError())]
        folder = Path(self.temp.name) / "cache/generated/audio"
        for case in cases:
            with self.subTest(case=case), patch.object(backend, "_AUDIO_MAX_BYTES", 4), patch.object(
                    backend, "is_safe_url", return_value=True), patch.object(
                    backend, "create_ssrf_safe_async_client", return_value=self.audio_download(**case)):
                with self.assertRaises((ValueError, OSError, backend.asyncio.CancelledError)):
                    REAL_SAVE_AUDIO("https://replicate.delivery/test")
            self.assertEqual(list(folder.glob("*")), [])

    def test_audio_blocks_unsafe_url_before_client(self):
        with patch.object(backend, "create_ssrf_safe_async_client") as client:
            for url in ("http://example.com/a", "https://user:password@example.com/a"):
                with self.assertRaises(backend.ReplicateError):
                    REAL_SAVE_AUDIO(url)
            with patch.object(backend, "is_safe_url", return_value=False):
                with self.assertRaises(ValueError):
                    REAL_SAVE_AUDIO("https://127.0.0.1/a")
            client.assert_not_called()

    def test_public_guard_blocks_private_connect_address(self):
        # Exercise the real public guard, not a replacement HTTP module.
        import socket
        async def probe():
            async with backend.create_ssrf_safe_async_client(trust_env=False, timeout=1) as client:
                with patch("socket.getaddrinfo", return_value=[
                        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))]):
                    with self.assertRaisesRegex(Exception, "Blocked request to cloud metadata address during connect"):
                        await client.get("https://replicate.delivery/test")
        backend.asyncio.run(probe())

    def test_audio_total_deadline_removes_partial(self):
        client = self.audio_download()
        download = client.stream.return_value.__aenter__.return_value
        async def stall(**kwargs):
            yield b"partial"
            await backend.asyncio.sleep(60)
        download.aiter_bytes = stall
        real_wait = backend.asyncio.wait_for
        async def short_wait(coro, *, timeout):
            self.assertEqual(timeout, 60)
            return await real_wait(coro, timeout=0.01)
        with patch.object(backend, "is_safe_url", return_value=True), patch.object(
                backend, "create_ssrf_safe_async_client", return_value=client), patch.object(
                backend.asyncio, "wait_for", side_effect=short_wait):
            with self.assertRaises(TimeoutError):
                REAL_SAVE_AUDIO("https://replicate.delivery/test")
        folder = Path(self.temp.name) / "cache/generated/audio"
        self.assertEqual(list(folder.glob("*")), [])

    def test_transcript_bound_profile_and_empty_text(self):
        path = backend.save_transcript("")
        self.assertEqual(path.read_bytes(), b"")
        self.assertTrue(path.is_relative_to(Path(self.temp.name) / "cache/generated/transcripts"))
        with patch.object(backend, "_TRANSCRIPT_MAX_BYTES", 3):
            for text in ("abcd", "éé"):
                with self.assertRaises(ValueError):
                    backend.save_transcript(text)
        self.assertEqual(list(path.parent.glob("*")), [path])

    def test_real_tts_output_copy_and_native_dispatch(self):
        from tools.tts_tool_plugins import _dispatch_to_plugin_provider
        source = Path(self.temp.name) / "cached.mp3"
        source.write_bytes(b"offline-speech")
        self.audio.return_value = source
        target = Path(self.temp.name) / "spoken.mp3"
        with patch.object(backend.shutil, "copyfile", wraps=shutil_copyfile), patch(
                "tools.tts_tool_plugins._lookup_plugin_provider", return_value=self.tts):
            result = _dispatch_to_plugin_provider("hello", str(target), "replicate",
                {"model": backend.TTS_MODEL, "voice": "English_Wiselady", "speed": 1.0})
        self.assertEqual(result, str(target))
        self.assertEqual(target.read_bytes(), b"offline-speech")

    def test_real_stt_native_dispatch(self):
        from tools.transcription_command import _dispatch_to_plugin_provider
        self.post.return_value = response({"transcription": "native transcript"})
        with patch("hermes_cli.plugins._ensure_plugins_discovered"), patch(
                "agent.transcription_registry.get_provider", return_value=self.stt):
            result = _dispatch_to_plugin_provider(str(self.file), "replicate", model=backend.STT_MODEL,
                                                  language="en", prompt=None)
        assert result is not None
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "native transcript")


if __name__ == "__main__":
    unittest.main()
