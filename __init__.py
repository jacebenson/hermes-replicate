"""Native Hermes image, video, transcription and speech providers for Replicate."""
from __future__ import annotations

import asyncio
import re
import math
from collections import OrderedDict
from copy import deepcopy
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry
import base64
import tempfile
from hermes_constants import get_hermes_home
from tools.url_safety import is_safe_url, create_ssrf_safe_async_client
import json
import mimetypes
from pathlib import Path
import shutil
import time
from urllib.parse import quote, urlparse

import requests
from agent.secret_scope import get_secret
from agent.image_gen_provider import (ImageGenProvider, DEFAULT_ASPECT_RATIO,
    resolve_aspect_ratio, save_url_image, success_response, error_response)
from agent.video_gen_provider import VideoGenProvider, save_url_video
from agent import video_gen_provider as video_api
from agent.tts_provider import TTSProvider
from agent.transcription_provider import TranscriptionProvider

# Snapshots supply convenience preset names only. Runtime schemas always come from the API.
CATALOG = json.loads((Path(__file__).parent / "docs/model-schemas.json").read_text())
IMAGE_MODELS = tuple(CATALOG)[:12]
DEFAULT_MODEL = "black-forest-labs/flux-schnell"
ANIMAGINE = "aisha-ai-official/animagine-xl-v4-opt"
VIDEO_MODEL = "wan-video/wan-2.2-t2v-fast"
STT_MODEL = "openai/whisper"
TTS_MODEL = "minimax/speech-02-turbo"
_VERSION = CATALOG[ANIMAGINE]["version"]
_API = "https://api.replicate.com/v1/predictions"
_SIZES = {"landscape": (1344, 768), "square": (1024, 1024), "portrait": (768, 1344)}
_RATIOS = {"landscape": "16:9", "square": "1:1", "portrait": "9:16"}
_POLL_TIMEOUT = 300.0
_POLL_INTERVAL = 1.5


class ReplicateError(RuntimeError):
    def __init__(self, message, kind="api_error"):
        super().__init__(message)
        self.kind = kind


from typing import NoReturn


def invalid(message) -> NoReturn:
    raise ReplicateError(message, "invalid_input")


def require_text(value, label="prompt"):
    if not isinstance(value, str) or not value.strip():
        invalid(f"A non-empty {label} is required")
    return value.strip()


def selected_model(modality, explicit, default):
    # Native tool dispatch supplies the user-configured model, never an LLM choice.
    if explicit is not None:
        return explicit
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly() or {}
    section = config.get(modality, {})
    nested = section.get("replicate", {})
    settings = config.get("plugins", {}).get("entries", {}).get("replicate", {}).get("settings", {})
    key = {"image_gen": "image_model", "video_gen": "video_model", "stt": "stt_model", "tts": "tts_model"}[modality]
    return nested.get("model") or settings.get(key) or default


_IDENTIFIER = re.compile(r"([a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*)(?::([0-9a-f]{64}))?", re.ASCII)
_SCHEMA_CACHE = OrderedDict()
_SCHEMA_CACHE_SIZE = 32
_SCHEMA_TTL = 300


def model_identifier(model, version=None):
    match = _IDENTIFIER.fullmatch(model) if isinstance(model, str) else None
    if not match:
        invalid("Model must be owner/name or owner/name:64-lowercase-hex-version")
    if version is not None and (not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version)):
        invalid("Pinned version must be 64 lowercase hexadecimal characters")
    if version and match[2] and version != match[2]:
        invalid("Conflicting model identifier and configured pinned version")
    return match[1], version or match[2]


def credential():
    token = (get_secret("REPLICATE_API_TOKEN") or "").strip()
    if not token:
        raise ReplicateError("Configure REPLICATE_API_TOKEN for the active Hermes profile", "missing_api_key")
    return token


def discover(model):
    name, version = model_identifier(model)
    token = credential()  # A cached schema never bypasses profile secret checks.
    key = (str(get_hermes_home().resolve()), name, version)
    now = time.monotonic()
    cached = _SCHEMA_CACHE.get(key)
    if cached and now - cached[0] < _SCHEMA_TTL:
        _SCHEMA_CACHE.move_to_end(key)
        return deepcopy(cached[1])
    url = f"https://api.replicate.com/v1/models/{name}"
    if version:
        url += f"/versions/{version}"
    try:
        response = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                                timeout=30, allow_redirects=False)
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise ValueError()
        data = response.json()
        is_official = not version and data.get("is_official") is True
        data = data if version else data.get("latest_version")
        if not isinstance(data, dict) or not re.fullmatch(r"[0-9a-f]{64}", data.get("id", "")):
            raise ValueError()
        if version and data["id"] != version:
            raise ValueError()
        schemas = data["openapi_schema"]["components"]["schemas"]
        if not isinstance(schemas.get("Input"), dict) or not isinstance(schemas.get("Output"), dict):
            raise ValueError()
        # No external schema resolution or unbounded schema responses.
        if len(json.dumps(schemas)) > 2 * 1024 * 1024:
            raise ValueError()
        def check(value):
            if isinstance(value, dict):
                if "$ref" in value and not value["$ref"].startswith("#/components/schemas/"):
                    raise ValueError()
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)
        check(schemas)
        result = {"version": data["id"], "schemas": schemas, "is_official": is_official}
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ReplicateError(f"Model schema discovery HTTP {status}; check model access and identifier", "schema_error") from None
    except requests.RequestException:
        raise ReplicateError("Model schema discovery failed; retry without changing models", "schema_error") from None
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise ReplicateError("Model has no supported input/output schema", "schema_error") from None
    _SCHEMA_CACHE[key] = (now, result)
    _SCHEMA_CACHE.move_to_end(key)
    while len(_SCHEMA_CACHE) > _SCHEMA_CACHE_SIZE:
        _SCHEMA_CACHE.popitem(last=False)
    return deepcopy(result)


def validate_input(model, payload, metadata=None):
    metadata = metadata or discover(model)
    schemas = metadata["schemas"]
    schema = {**schemas["Input"], "components": {"schemas": schemas}, "additionalProperties": False}
    try:
        validator = Draft202012Validator(schema, registry=Registry(), format_checker=FormatChecker())
        errors = list(validator.iter_errors(payload))
        if errors:
            error = errors[0]
            field = ".".join(map(str, error.absolute_path)) or "Input"
            # Do not echo values, which can contain audio, prompts or secrets.
            detail = str(error.validator_value) if error.validator in {"required", "enum", "minimum", "maximum", "type"} else "schema constraint"
            invalid(f"Model input {field} violates {error.validator}: {detail}. Configure input_mapping or extra_inputs")
        json.dumps(payload, allow_nan=False)
    except ReplicateError:
        raise
    except Exception:
        invalid("Unsupported model schema or non-JSON inputs; configure a compatible model")


def model_config(modality, model):
    from hermes_cli.config import load_config_readonly
    config = load_config_readonly() or {}
    section = config.get(modality, {}).get("replicate", {})
    settings = config.get("plugins", {}).get("entries", {}).get("replicate", {}).get("settings", {})
    prefix = {"image_gen": "image", "video_gen": "video", "stt": "stt", "tts": "tts"}[modality]
    options = settings.get(prefix + "_config", {})
    if isinstance(options, str):
        try:
            options = json.loads(options)
        except ValueError:
            invalid(f"{prefix}_config must be a JSON object")
    if not isinstance(options, dict) or set(options) - {"version", "input_mapping", "output_mapping", "extra_inputs", "prediction_route"}:
        invalid("Model config accepts only version, input_mapping, output_mapping, extra_inputs, prediction_route")
    options = {**options, **{k: v for k, v in section.items() if k != "model"}}
    if set(options) - {"version", "input_mapping", "output_mapping", "extra_inputs", "prediction_route"}:
        invalid("Unknown Replicate model configuration field")
    name, version = model_identifier(model, options.get("version"))
    route = options.get("prediction_route", "auto")
    if route not in ("auto", "official_name"):
        invalid("prediction_route must be auto or official_name")
    if route == "official_name" and (version or modality not in ("image_gen", "video_gen")):
        invalid("official_name requires an unpinned image or video model")
    return name + (":" + version if version else ""), options


_ALIASES = {"prompt": ("prompt",), "text": ("text", "input_text"),
    "audio": ("audio", "audio_file"), "voice": ("voice_id", "voice"),
    "format": ("audio_format", "format"), "instructions": ("instructions",),
    "initial_prompt": ("initial_prompt",), "language": ("language",),
    "speed": ("speed",), "duration": ("duration",), "seed": ("seed",),
    "aspect_ratio": ("aspect_ratio",), "resolution": ("resolution",),
    "width": ("width",), "height": ("height",)}


def prepare(modality, model, values):
    model, options = model_config(modality, model)
    metadata = discover(model)
    if options.get("prediction_route") == "official_name" and metadata.get("is_official") is not True:
        invalid("official_name requires an official model")
    properties = metadata["schemas"]["Input"].get("properties", {})
    mapping = options.get("input_mapping", {})
    extras = options.get("extra_inputs", {})
    if not isinstance(mapping, dict) or not isinstance(extras, dict):
        invalid("input_mapping and extra_inputs must be JSON objects")
    if set(mapping) - set(_ALIASES) or any(not isinstance(v, str) or v not in properties for v in mapping.values()):
        invalid("input_mapping must map supported native names to existing schema fields")
    if len(set(mapping.values())) != len(mapping):
        invalid("input_mapping destinations must be unique")
    def safe_extras(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or any(word in key.lower() for word in ("secret", "token", "password", "authorization", "api_key", "endpoint", "webhook")):
                    invalid("extra_inputs cannot configure endpoints, webhooks or credentials")
                safe_extras(child)
        elif isinstance(value, list):
            for child in value:
                safe_extras(child)
    safe_extras(extras)
    payload = dict(extras)
    for source, value in values.items():
        if value is None:
            continue
        candidates = [mapping[source]] if source in mapping else [k for k in _ALIASES[source] if k in properties]
        if len(candidates) > 1:
            invalid(f"Ambiguous {source}; configure input_mapping")
        if candidates:
            key = candidates[0]
            if key in payload:
                if source in {"prompt", "text", "audio"}:
                    invalid(f"extra_inputs conflicts with native field {key}")
                continue  # Explicit configured model inputs override convenience defaults.
            payload[key] = value
        elif source in {"prompt", "text", "audio"} or source in mapping:
            invalid(f"No model field for {source}; configure input_mapping")
        elif source not in {"width", "height", "aspect_ratio", "resolution", "format"}:
            invalid(f"Model does not support {source}; remove the option or configure input_mapping")
    # Existing preset-specific constants are convenience defaults, never schema substitutes.
    if model.split(":")[0] == VIDEO_MODEL:
        for k, v in {"num_frames": 81, "frames_per_second": 16}.items():
            if k in properties:
                payload.setdefault(k, v)
    if model.split(":")[0] == STT_MODEL and "transcription" in properties:
        payload.setdefault("transcription", "plain text")
    validate_input(model, payload, metadata)
    validate_output_mapping(metadata, options, modality)
    return model, payload, metadata, options


def validate_output_mapping(metadata, options, modality=None):
    path = options.get("output_mapping")
    if path is not None and (not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", path)):
        invalid("output_mapping must be a dot-separated object key/list index path")
    schema = metadata["schemas"]["Output"]
    if path:
        for part in path.split("."):
            seen = set()
            while "$ref" in schema:
                ref = schema["$ref"]
                if ref in seen or len(seen) >= 16:
                    invalid("Recursive output schema is unsupported")
                seen.add(ref)
                schema = metadata["schemas"].get(ref.rsplit("/", 1)[-1], {})
            if schema.get("type") == "array" and part.isdigit():
                schema = schema.get("items", {})
            elif part in schema.get("properties", {}):
                schema = schema["properties"][part]
            else:
                invalid("output_mapping path is not present in model output schema")
    for _ in range(16):
        if "$ref" in schema:
            schema = metadata["schemas"].get(schema["$ref"].rsplit("/", 1)[-1], {})
        elif schema.get("type") == "array":
            schema = schema.get("items", {})
        elif schema.get("type") == "object" or "properties" in schema:
            keys = ("transcription", "transcript", "text") if modality == "stt" else ("image", "images", "video", "audio", "url", "output")
            matches = [k for k in keys if k in schema.get("properties", {})]
            if len(matches) != 1:
                invalid("Unsupported or ambiguous output schema; configure output_mapping")
            schema = schema["properties"][matches[0]]
        elif schema.get("type") == "string":
            return
        else:
            invalid("Unsupported output schema; configure output_mapping to a string or media URL")
    invalid("Recursive output schema is unsupported")


def mapped_output(output, options, transcript=False):
    path = options.get("output_mapping")
    try:
        if path:
            for part in path.split("."):
                output = output[int(part)] if isinstance(output, list) else output[part]
        elif isinstance(output, dict):
            keys = ("transcription", "transcript", "text") if transcript else ("image", "images", "video", "audio", "url", "output")
            matches = [key for key in keys if key in output]
            if len(matches) != 1:
                raise ValueError()
            output = output[matches[0]]
        if isinstance(output, list) and output:
            output = output[0]
        if isinstance(output, dict):
            return mapped_output(output, {}, transcript)
        if transcript and not isinstance(output, str):
            raise ValueError()
        return output
    except (ValueError, TypeError, KeyError, IndexError):
        raise ReplicateError("Unsupported or ambiguous model output; configure output_mapping", "invalid_response") from None


def predict(model, payload, metadata=None, *, model_endpoint=False, prediction_route="auto"):
    """One creation, bounded polling; never echo server bodies, URLs or exceptions."""
    metadata = metadata or discover(model)
    validate_input(model, payload, metadata)
    token = credential()
    if not token:
        raise ReplicateError("Configure REPLICATE_API_TOKEN for the active Hermes profile", "missing_api_key")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    deadline = time.monotonic() + _POLL_TIMEOUT

    def remaining():
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise requests.Timeout()
        return min(30.0, budget)

    def read(response):
        response.raise_for_status()
        if not 200 <= response.status_code < 300:
            raise ValueError()
        data = response.json()
        if not isinstance(data, dict) or data.get("status") not in {"starting", "processing", "succeeded", "failed", "canceled"}:
            raise ValueError()
        if not isinstance(data.get("id"), str) or not data["id"].strip():
            raise ValueError()
        return data

    # Official image/video models use their stable model API, not hidden versions.
    name, pinned = model_identifier(model)
    url = _API
    body = {"version": metadata["version"], "input": payload}
    if prediction_route not in ("auto", "official_name"):
        invalid("prediction_route must be auto or official_name")
    if prediction_route == "official_name":
        if not model_endpoint or pinned or metadata.get("is_official") is not True:
            invalid("official_name requires an unpinned official image or video model")
        body = {"version": name, "input": payload}
    elif model_endpoint and not pinned and metadata.get("is_official") is True:
        url = f"https://api.replicate.com/v1/models/{name}/predictions"
        body = {"input": payload}
    try:
        prediction = read(requests.post(url, headers=headers, timeout=remaining(),
            allow_redirects=False, json=body))
        prediction_id = prediction["id"]
        while prediction["status"] in {"starting", "processing"}:
            time.sleep(min(_POLL_INTERVAL, remaining()))
            prediction = read(requests.get(f"{_API}/{quote(prediction_id, safe='')}",
                headers=headers, timeout=remaining(), allow_redirects=False))
        if prediction["status"] != "succeeded":
            raise ReplicateError("Replicate prediction did not succeed")
        return prediction.get("output")
    except requests.Timeout:
        raise ReplicateError("Replicate timed out; remote work may still be billed", "timeout") from None
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ReplicateError(f"Replicate HTTP error ({status})", "api_error") from None
    except requests.RequestException:
        raise ReplicateError("Could not communicate with Replicate", "connection_error") from None
    except (ValueError, TypeError):
        raise ReplicateError("Invalid Replicate prediction response", "invalid_response") from None


def output_url(output):
    if isinstance(output, list):
        if not output:
            raise ReplicateError("Replicate returned no media", "empty_response")
        output = output[0]
    if not isinstance(output, str):
        raise ReplicateError("Invalid Replicate output", "invalid_response")
    parsed = urlparse(output)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ReplicateError("Invalid Replicate output URL", "invalid_response")
    return output


def persist(output, saver):
    url = output_url(output)
    try:
        return str(saver(url))
    except Exception:
        raise ReplicateError("Prediction succeeded but caching failed; retrying may incur another charge", "cache_error") from None


_AUDIO_MAX_BYTES = 50 * 1024 * 1024
_TRANSCRIPT_MAX_BYTES = 20 * 1024 * 1024
_AUDIO_TYPES = {"audio/mpeg": "mp3", "audio/wav": "wav", "audio/x-wav": "wav", "audio/flac": "flac"}


def cache_file(kind, extension):
    """Exclusive private file in the active profile; arguments are local constants."""
    directory = get_hermes_home() / "cache" / "generated" / kind
    directory.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(mode="wb", prefix="replicate_", suffix="." + extension,
                                       dir=directory, delete=False)


def save_transcript(text):
    # Bound both encoding work and the persisted UTF-8 representation.
    if len(text) > _TRANSCRIPT_MAX_BYTES:
        raise ValueError("Transcript exceeds cache limit")
    raw = text.encode("utf-8")
    if len(raw) > _TRANSCRIPT_MAX_BYTES:
        raise ValueError("Transcript exceeds cache limit")
    file = cache_file("transcripts", "txt")
    path = Path(file.name)
    try:
        with file:
            file.write(raw)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


async def download_audio(url):
    # Public guarded transport validates the actual connect address, preventing
    # DNS rebinding. Reject redirects altogether; never forward credentials.
    url = output_url(url)
    if not is_safe_url(url):
        raise ValueError("Unsafe audio URL")
    async with create_ssrf_safe_async_client(timeout=60, follow_redirects=False,
                                             trust_env=False) as client:
        async with client.stream("GET", url, headers=None) as response:
            if not 200 <= response.status_code < 300:
                raise ValueError("Audio download failed")
            mime = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if mime not in _AUDIO_TYPES:
                raise ValueError("Unexpected audio content type")
            length = response.headers.get("Content-Length")
            if length is not None and not 0 < int(length) <= _AUDIO_MAX_BYTES:
                raise ValueError("Invalid audio length")
            file = cache_file("audio", _AUDIO_TYPES[mime])
            path = Path(file.name)
            try:
                size = 0
                with file:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        size += len(chunk)
                        if size > _AUDIO_MAX_BYTES:
                            raise ValueError("Audio exceeds cache limit")
                        file.write(chunk)
                    if not size:
                        raise ValueError("Audio was empty")
                return path
            except BaseException:
                path.unlink(missing_ok=True)
                raise


def save_audio(url):
    async def bounded():
        return await asyncio.wait_for(download_audio(url), timeout=60)
    return asyncio.run(bounded())


class Metadata:
    models: tuple[str, ...] = ()
    @property
    def name(self):
        return "replicate"

    @property
    def display_name(self):
        return "Replicate"

    def is_available(self):
        return bool((get_secret("REPLICATE_API_TOKEN") or "").strip())

    def default_model(self):
        return self.models[0]

    def list_models(self):
        return [{"id": m, "display": m} for m in self.models]

    def get_setup_schema(self):
        return {"name": "Replicate", "badge": "paid", "tag": "Replicate paid inference",
            "env_vars": [{"key": "REPLICATE_API_TOKEN", "prompt": "Replicate API token",
                          "url": "https://replicate.com/account/api-tokens"}]}


class ReplicateImageGenProvider(Metadata, ImageGenProvider):
    models = IMAGE_MODELS

    def capabilities(self):
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(self, prompt, aspect_ratio=DEFAULT_ASPECT_RATIO, *, image_url=None,
                 reference_image_urls=None, **kwargs):
        aspect = resolve_aspect_ratio(aspect_ratio)
        model = DEFAULT_MODEL
        clean = ""
        try:
            clean = require_text(prompt)
            if image_url or reference_image_urls or kwargs.get("options"):
                invalid("Reference images and model-specific options are unsupported")
            model = selected_model("image_gen", kwargs.get("model"), DEFAULT_MODEL)
            width, height = _SIZES[aspect]
            model, payload, metadata, options = prepare("image_gen", model,
                {"prompt": clean, "aspect_ratio": _RATIOS[aspect], "width": width, "height": height})
            path = persist(mapped_output(predict(model, payload, metadata, model_endpoint=True,
                                               prediction_route=options.get("prediction_route", "auto")), options),
                           lambda url: save_url_image(url, prefix="replicate", timeout=30))
            return success_response(image=path, model=model, prompt=clean,
                aspect_ratio=aspect, provider=self.name, modality="text")
        except ReplicateError as exc:
            return error_response(error=str(exc), error_type=exc.kind, provider=self.name,
                model=model, prompt=clean, aspect_ratio=aspect)
        except Exception:
            return error_response(error="Could not read Replicate configuration", error_type="configuration_error", provider=self.name)


class ReplicateVideoGenProvider(Metadata, VideoGenProvider):
    models = (VIDEO_MODEL,)

    def capabilities(self):
        return {"modalities": ["text"], "aspect_ratios": ["16:9", "9:16"],
            "resolutions": ["480p", "720p"], "min_duration": 5, "max_duration": 5,
            "supports_audio": False, "supports_seed": True,
            "supports_negative_prompt": False, "supports_upscale": False, "max_reference_images": 0}

    def generate(self, prompt, *, model=None, image_url=None, reference_image_urls=None,
                 duration=None, aspect_ratio="16:9", resolution="720p", negative_prompt=None,
                 audio=None, seed=None, **kwargs):
        clean = ""
        try:
            clean = require_text(prompt)
            model = selected_model("video_gen", model, VIDEO_MODEL)
            if image_url or reference_image_urls or negative_prompt or audio or kwargs.get("upscale") or kwargs.get("options"):
                invalid("References, negative prompts, audio and upscale require model-specific extra_inputs")
            model, payload, metadata, options = prepare("video_gen", model,
                {"prompt": clean, "aspect_ratio": aspect_ratio, "resolution": resolution,
                 "duration": duration, "seed": seed})
            path = persist(mapped_output(predict(model, payload, metadata, model_endpoint=True,
                                               prediction_route=options.get("prediction_route", "auto")), options),
                           lambda url: save_url_video(url, prefix="replicate", timeout=60))
            if duration is None and model.split(":")[0] == VIDEO_MODEL:
                properties = metadata["schemas"]["Input"].get("properties", {})
                frames = payload.get("num_frames", properties.get("num_frames", {}).get("default"))
                fps = payload.get("frames_per_second", properties.get("frames_per_second", {}).get("default"))
                if all(type(value) in (int, float) and math.isfinite(value) and value > 0
                       for value in (frames, fps)):
                    # Configured duration, not a measurement of interpolated output.
                    duration = frames / fps
            result = video_api.success_response(video=path, model=model, prompt=clean,
                aspect_ratio=aspect_ratio, provider=self.name)
            # Hermes' helper coerces durations to integer seconds and None to zero.
            result["duration"] = duration
            return result
        except ReplicateError as exc:
            return video_api.error_response(error=str(exc), error_type=exc.kind, provider=self.name,
                model=model or VIDEO_MODEL, prompt=clean, aspect_ratio=aspect_ratio)
        except Exception:
            return video_api.error_response(error="Invalid Replicate configuration", error_type="configuration_error", provider=self.name)


class ReplicateTranscriptionProvider(Metadata, TranscriptionProvider):
    models = (STT_MODEL,)

    def transcribe(self, file_path, *, model=None, language=None, **extra):
        try:
            model = selected_model("stt", model, STT_MODEL)
            model, payload, metadata, options = prepare("stt", model,
                {"audio": "", "language": language,
                 "initial_prompt": extra.get("prompt")})
            path = Path(file_path)
            if not path.is_file() or not 0 < path.stat().st_size <= 20 * 1024 * 1024:
                invalid("Audio must be an existing nonempty file of at most 20 MiB")
            mime = mimetypes.guess_type(path.name)[0]
            if not mime or not mime.startswith(("audio/", "video/")):
                invalid("Unsupported audio file extension")
            raw = path.read_bytes()
            if not 0 < len(raw) <= 20 * 1024 * 1024:
                invalid("Audio exceeds upload limit")
            audio_key = options.get("input_mapping", {}).get("audio") or next(k for k in _ALIASES["audio"] if k in payload)
            payload[audio_key] = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
            transcript = mapped_output(predict(model, payload, metadata), options, transcript=True)
            # Durable transcript in the profile cache; native dispatcher consumes transcript.
            saved = save_transcript(transcript)
            return {"success": True, "transcript": transcript, "provider": self.name, "path": str(saved)}
        except ReplicateError as exc:
            return {"success": False, "transcript": "", "provider": self.name, "error": str(exc), "error_type": exc.kind}
        except Exception:
            return {"success": False, "transcript": "", "provider": self.name,
                    "error": "Could not read audio, configuration or persist transcript", "error_type": "local_error"}


class ReplicateTTSProvider(Metadata, TTSProvider):
    models = (TTS_MODEL,)

    def list_voices(self):
        return [{"id": "English_Wiselady", "display": "English Wiselady"}]

    def synthesize(self, text, output_path, *, voice=None, model=None, speed=None, format="mp3", **extra):
        try:
            clean = require_text(text, "text")
            if len(clean) > 10000:
                invalid("Speech text exceeds 10,000 characters")
            model = selected_model("tts", model, TTS_MODEL)
            if format not in ("mp3", "wav", "flac"):
                invalid("Supported speech formats: mp3, wav, flac")
            model, payload, metadata, options = prepare("tts", model,
                {"text": clean, "voice": voice or (self.default_voice() if model.split(":")[0] == TTS_MODEL else None),
                 "speed": speed, "format": format, "instructions": extra.get("instructions")})
            path = persist(mapped_output(predict(model, payload, metadata), options), save_audio)
            target = Path(output_path).with_suffix("." + format)
            if Path(path).resolve() != target.resolve():
                shutil.copyfile(path, target)
            return str(target)
        except ReplicateError:
            raise
        except Exception:
            raise ReplicateError("Could not read configuration or save speech output", "local_error") from None


def register(ctx):
    ctx.register_image_gen_provider(ReplicateImageGenProvider())
    ctx.register_video_gen_provider(ReplicateVideoGenProvider())
    ctx.register_transcription_provider(ReplicateTranscriptionProvider())
    ctx.register_tts_provider(ReplicateTTSProvider())
