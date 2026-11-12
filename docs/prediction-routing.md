# Image/video prediction routing

Replicate's [official model API documentation](https://replicate.com/docs/topics/models/official-models#api)
specifies `POST /v1/models/{owner}/{name}/predictions` with an `input` body and
no version for official models. The model metadata's `latest_version.id` remains
useful for schema validation; it is not the official prediction invocation contract.

The image/video providers now preserve the model metadata's strict boolean
`is_official` classification in the profile-scoped schema cache. Unpinned official
models use the model endpoint. Community models, metadata without an official
classification, and explicit version pins retain `POST /v1/predictions` with the
resolved version. Pins are never silently discarded. Speech and transcription
request routing is unchanged by this targeted correction.

Both routes still validate inputs before creation, send exactly one creation POST,
refuse redirects, sanitize errors, and poll the fixed prediction endpoint by ID.
There is no fallback POST or automatic creation retry: a failed response may have
an unknown billing outcome.

## Explicit official-name invocation

The [HTTP predictions.create contract](https://replicate.com/docs/reference/http#predictions.create-request-body-version)
also supports `POST /v1/predictions` with `version: "owner/name"` for official
models. This is a documented alternative to the model endpoint, not a reason to
switch all existing working calls. Image/video configuration accepts
`prediction_route: official_name` to select it before creation. The default
`auto` preserves the existing official model endpoint and community/pinned routing.

For example, the native `video_gen.replicate` configuration can contain:

```yaml
prediction_route: official_name
```

The same field is accepted in plugin `video_config` or `image_config` JSON.
Only unpinned models whose metadata reports `is_official: true` may select this
route. Community models, explicit pins, other route values and speech modalities
are rejected rather than silently discarding a pin or choosing another endpoint.
Schema and input/output mapping validation still run. A failure never triggers
a second creation or automatic fallback. No model names are hardcoded in routing.

## Evidence and limits

The reported Flux Schnell and Wan 2.2 failures each had successful model metadata
reads (`is_official: true`), followed by a generic version-based creation POST that
returned HTTP 500 without a prediction ID. This establishes a request-contract
mismatch, not proof that it was the provider's only cause of failure.

`python3 -m unittest discover -s tests -p test_prediction_contract.py -v`
exercises actual provider generation, schema discovery, validation, request building,
and output handling using installed Hermes APIs, with HTTP and media persistence
mocked. The endpoint assertions failed for both image and video before the fix.
Expected requests come from the published API contract, not from a mock that merely
returns success for a changed endpoint. The full offline suite is run with:

```
python3 -m unittest discover -s tests -v
```

No paid prediction is needed to run these tests. They prove the corrected client
contract, not hosted generation or artifact delivery. Live image/video acceptance
requires separately authorized, bounded provider calls after review and deployment.
