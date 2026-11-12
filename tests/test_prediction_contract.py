"""Offline HTTP contracts from Replicate's official-model API documentation."""
import unittest
from unittest.mock import patch
import requests
import test_provider as original

backend = original.backend


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(PredictionContractTests(n) for n in PredictionContractTests.__dict__ if n.startswith('test_'))


class PredictionContractTests(original.ProviderTests):
    def metadata_get(self, url, **kwargs):
        model = url.split('/models/', 1)[1].split('/versions/', 1)[0]
        item = backend.CATALOG[model]
        data = {'id': item['version'], 'openapi_schema': {'components': {'schemas': item['schemas']}}}
        result = original.response()
        result.json.return_value = data if '/versions/' in url else {'is_official': self.official, 'latest_version': data}
        return result

    def test_official_image_and_video_use_model_endpoint(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        for provider, model in [(self.image, backend.DEFAULT_MODEL), (self.video, backend.VIDEO_MODEL)]:
            with self.subTest(model=model):
                self.post.reset_mock()
                result = provider.generate('contract', model=model)
                self.assertTrue(result['success'], result)
                self.post.assert_called_once()
                self.assertEqual(self.post.call_args.args[0], f'https://api.replicate.com/v1/models/{model}/predictions')
                self.assertEqual(set(self.post.call_args.kwargs['json']), {'input'})
                self.assertEqual(self.post.call_args.kwargs['json']['input']['prompt'], 'contract')
                self.assertFalse(self.post.call_args.kwargs['allow_redirects'])

    def test_community_and_explicit_pins_keep_version_endpoint(self):
        self.official = False
        self.get.side_effect = self.metadata_get
        for model in [backend.DEFAULT_MODEL, backend.DEFAULT_MODEL + ':' + backend.CATALOG[backend.DEFAULT_MODEL]['version']]:
            self.post.reset_mock()
            result = self.image.generate('contract', model=model)
            self.assertTrue(result['success'], result)
            self.post.assert_called_once()
            self.assertEqual(self.post.call_args.args[0], backend._API)
            self.assertEqual(self.post.call_args.kwargs['json']['version'], backend.CATALOG[backend.DEFAULT_MODEL]['version'])

    def test_official_configured_pin_keeps_exact_version(self):
        version = backend.CATALOG[backend.DEFAULT_MODEL]['version']
        self.official = True
        self.get.side_effect = self.metadata_get
        config = {'image_gen': {'replicate': {'version': version}}}
        with patch('hermes_cli.config.load_config_readonly', return_value=config):
            result = self.image.generate('contract')
        self.assertTrue(result['success'], result)
        self.assertTrue(self.get.call_args.args[0].endswith('/versions/' + version))
        self.assertEqual(self.post.call_args.args[0], backend._API)
        self.assertEqual(self.post.call_args.kwargs['json']['version'], version)

    def test_cached_official_classification_keeps_model_route(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        for _ in range(2):
            self.assertTrue(self.image.generate('contract')['success'])
            self.assertEqual(self.post.call_args.args[0], f'https://api.replicate.com/v1/models/{backend.DEFAULT_MODEL}/predictions')
        self.assertEqual(self.get.call_count, 1)
        self.assertEqual(self.post.call_count, 2)

    def test_removed_route_knob_rejected_before_creation(self):
        config = {'video_gen': {'replicate': {'prediction_route': 'official_name'}}}
        with patch('hermes_cli.config.load_config_readonly', return_value=config):
            self.assert_failure(lambda: self.video.generate('contract'), 'invalid_input')
        self.post.assert_not_called()

    def test_default_native_video_handler_payload(self):
        import json
        import tools.video_generation_tool as tool
        self.official = True
        self.get.side_effect = self.metadata_get
        with patch.object(tool, '_read_configured_video_provider', return_value='replicate'), patch.object(
                tool, '_resolve_active_provider', return_value=self.video), patch.object(
                tool, '_read_configured_video_model', return_value=None):
            result = json.loads(tool._handle_video_generate({'prompt': 'contract'}))
        self.assertTrue(result['success'], result)
        self.assertEqual(result['model'], 'prunaai/p-video')
        self.assertEqual(result['duration'], 5)
        self.post.assert_called_once()
        self.assertEqual(self.post.call_args.args[0],
                         'https://api.replicate.com/v1/models/prunaai/p-video/predictions')
        self.assertEqual(self.post.call_args.kwargs['json'], {'input': {
            'prompt': 'contract', 'aspect_ratio': '16:9', 'resolution': '720p',
            'duration': 5, 'draft': True, 'fps': 24, 'prompt_upsampling': False}})
        self.assertFalse(self.post.call_args.kwargs['allow_redirects'])
        self.assertEqual(self.video.capabilities()['resolutions'], ['720p'])

    def test_pvideo_defaults_allow_explicit_configuration(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        config = {'video_gen': {'replicate': {'extra_inputs': {
            'duration': 4, 'draft': False, 'fps': 48, 'prompt_upsampling': True}}}}
        with patch('hermes_cli.config.load_config_readonly', return_value=config):
            result = self.video.generate('contract', duration=5)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['duration'], 4)
        self.assertEqual(self.post.call_args.kwargs['json']['input'], {
            'prompt': 'contract', 'aspect_ratio': '16:9', 'resolution': '720p',
            'duration': 4, 'draft': False, 'fps': 48, 'prompt_upsampling': True})

    def test_wan_explicit_selection_preserves_old_recipe(self):
        result = self.video.generate('contract', model='wan-video/wan-2.2-t2v-fast', resolution='480p')
        self.assertTrue(result['success'], result)
        self.assertEqual(self.post.call_args.kwargs['json']['input'], {
            'prompt': 'contract', 'aspect_ratio': '16:9', 'resolution': '480p',
            'num_frames': 81, 'frames_per_second': 16})
        self.assertEqual(result['duration'], 5.0625)

    def test_manifest_matches_runtime_defaults(self):
        from hermes_cli.plugins_settings import plugin_settings_fields
        fields = plugin_settings_fields('replicate', original.ROOT)
        video = next(f for f in fields if f['key'] == 'video_model')
        self.assertEqual(video['default'], self.video.default_model())
        self.assertIn('version: 1.1.1\n', (original.ROOT / 'plugin.yaml').read_text())
        self.assertEqual(self.video.default_model(), 'prunaai/p-video')

    def test_pvideo_explicit_pin_preserved(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        version = backend.CATALOG[backend.VIDEO_MODEL]['version']
        result = self.video.generate('contract', model=backend.VIDEO_MODEL + ':' + version)
        self.assertTrue(result['success'], result)
        self.post.assert_called_once()
        self.assertEqual(self.post.call_args.args[0], backend._API)
        self.assertEqual(self.post.call_args.kwargs['json']['version'], version)
        self.assertTrue(self.post.call_args.kwargs['json']['input']['draft'])

    def test_official_creation_failure_never_retries(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        failed = original.response()
        failed.status_code = 500
        failed.raise_for_status.side_effect = requests.HTTPError(response=failed)
        self.post.return_value = failed
        self.assert_failure(lambda: self.image.generate('contract'), 'api_error')
        self.post.assert_called_once()

    def test_official_polling_uses_prediction_id_not_response_url(self):
        self.official = True
        def get(url, **kwargs):
            if '/models/' in url:
                return self.metadata_get(url, **kwargs)
            self.assertEqual(url, backend._API + '/test-id')
            return original.response()
        self.get.side_effect = get
        self.post.return_value = original.response(status='processing')
        with patch.object(backend.time, 'sleep'):
            self.assertTrue(self.image.generate('contract')['success'])
        self.post.assert_called_once()
        self.assertEqual(self.get.call_count, 2)
