# Verification

## Current implementation

Plugin 1.1.0 registers native image, video, transcription and TTS providers under `replicate`. All four accept user-configured `owner/name` or `owner/name:64-lowercase-hex-version` identifiers. Presets are suggestions, not an allowlist. Free-text settings and JSON model configuration are declared in `plugin.yaml`.

Runtime schemas come from authenticated read-only model/version GET requests, not the shipped snapshot. Unpinned names resolve the latest version; each prediction submits that resolved version. Explicit pins persist selection across requests. The in-memory cache has a 300-second TTL and 32-entry LRU bound, scoped by active Hermes home, model name and requested pin. Cache hits still require credentials.

Discovery discloses the chosen model/version and authenticates to Replicate before a paid POST, including when later validation fails. Input mappings, extra inputs and output paths are checked against the discovered schema before prediction creation. Schema access proves neither inference compatibility nor entitlement to run a prediction. See the [configuration examples and security disclosure](../README.md).

## Executed offline checks

Run from the checkout with the Hermes Python environment:

```sh
python -m unittest discover -s tests -v
hermes plugins doctor . --ci
hermes plugins validate .
```

Current documentation review reran all three:

- 36 offline tests passed. HTTP schema reads, prediction creation and polling are mocked, not live inference.
- Doctor passed discovery, manifest parsing, import and registration. Zero custom tools and hooks is expected for provider registrations.
- Catalog validation passed without warnings. Host isolation remains an informational notice, not catalog acceptance.

Tests exercise arbitrary identifiers across all four adapters, strict identifiers, version pins and conflicts, profile-scoped cache TTL/bounds, required fields, enum/type/range validation, nonstandard input mappings and output paths, ambiguity, discovery failures, external-reference rejection, and missing credentials on cache hits. They also cover native dispatch, scoped secrets, settings rendering, persistence, sanitized failures and polling deadlines.

Audio tests cover the public SSRF-safe transport, rejection of private connection addresses, redirects and unsupported MIME types, byte limits, deadline cleanup and profile-private cache files. Transcript tests include empty text and UTF-8 bounds. These are local safety/contract checks, not proof of speech quality or remote model behavior.

## Live image results

The authorized image run attempted 13 distinct models and produced 12 local images. Seedream-3 failed remotely; a separate SDXL prediction supplied the twelfth successful model.

| Model | Recorded result |
| --- | --- |
| `black-forest-labs/flux-schnell` | Success |
| `black-forest-labs/flux-dev` | Success |
| `black-forest-labs/flux-1.1-pro` | Success |
| `black-forest-labs/flux-pro` | Success |
| `black-forest-labs/flux-kontext-pro` | Success, text-to-image |
| `ideogram-ai/ideogram-v3-turbo` | Success |
| `recraft-ai/recraft-v3` | Success |
| `bytedance/seedream-3` | Remote prediction status `failed`; no image |
| `aisha-ai-official/animagine-xl-v4-opt` | Success |
| `prunaai/z-image-turbo` | Success |
| `qwen/qwen-image` | Success |
| `black-forest-labs/flux-2-klein-4b` | Success |
| `stability-ai/sdxl` | Success, replacement showcase tile |

This is not the same set as the 12 image suggestions in the shipped snapshot. The three Google Imagen suggestions were not part of this live run. A failed Seedream prediction does not establish its root cause or permanent incompatibility.

The receipts establish individual generation results, not exhaustive aspect-ratio coverage, current pricing, output rights, or compatibility with every custom schema.

## Live video and speech results

- `wan-video/wan-2.2-t2v-fast` produced a locally saved H.264 video. FFprobe read 832×480 pixels and 5.066667 seconds. The original provider response incorrectly reported duration zero; this is fixed to report configured frame-count/fps duration, 5.0625 seconds for the default, distinct from the measured interpolated file duration. Unknown custom-model duration remains unset.
- `minimax/speech-02-turbo` produced a locally saved MP3. FFprobe read 7.883188 seconds.
- `openai/whisper` transcribed that generated MP3 as: "One plugin. Images, video, speech, and transcription. Powered by Replicate." This matches the synthesis input after trimming surrounding whitespace.

The [cover](assets/banner.png), [12-model gallery](assets/model-showcase.png), [model labels](assets/examples.json), [video](assets/video-demo.mp4), [speech](assets/speech-demo.mp3) and [transcript](assets/transcription-demo.txt) are saved examples from those live calls. The waveform uses the generated speech samples, not decorative random bars.

## Remaining checks

- Installed end-to-end selection and delivery through Hermes' user-facing interfaces, including custom mappings and pinned versions. Direct provider invocation is not a running Desktop UI test.
- Live checks for other models as needed. Read-only discovery and offline contracts must not be labeled as successful generation.

Streaming speech, low-latency conversations and native reference/editing workflows remain outside the implemented interface. Configured model-specific fields do not automatically add native UI capabilities.
