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

    def test_explicit_official_name_route_uses_generic_endpoint(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        for modality, provider, model in [('image_gen', self.image, backend.DEFAULT_MODEL),
                                          ('video_gen', self.video, backend.VIDEO_MODEL)]:
            with self.subTest(modality=modality):
                self.post.reset_mock()
                config = {modality: {'replicate': {'prediction_route': 'official_name'}}}
                with patch('hermes_cli.config.load_config_readonly', return_value=config):
                    result = provider.generate('contract', model=model)
                self.assertTrue(result['success'], result)
                self.post.assert_called_once()
                self.assertEqual(self.post.call_args.args[0], backend._API)
                self.assertEqual(set(self.post.call_args.kwargs['json']), {'version', 'input'})
                self.assertEqual(self.post.call_args.kwargs['json']['version'], model)
                self.assertEqual(self.post.call_args.kwargs['json']['input']['prompt'], 'contract')
                self.assertFalse(self.post.call_args.kwargs['allow_redirects'])

    def test_name_route_rejects_community_pins_and_unknown_routes_before_creation(self):
        for official, model, route in [
            (False, backend.DEFAULT_MODEL, 'official_name'),
            (True, backend.DEFAULT_MODEL + ':' + backend.CATALOG[backend.DEFAULT_MODEL]['version'], 'official_name'),
            (True, backend.DEFAULT_MODEL, 'https://example.com'),
        ]:
            with self.subTest(official=official, model=model, route=route):
                backend._SCHEMA_CACHE.clear()
                self.official = official
                self.get.side_effect = self.metadata_get
                config = {'image_gen': {'replicate': {'prediction_route': route}}}
                with patch('hermes_cli.config.load_config_readonly', return_value=config):
                    self.assert_failure(lambda: self.image.generate('contract', model=model), 'invalid_input')

    def test_name_route_failure_never_falls_back(self):
        self.official = True
        self.get.side_effect = self.metadata_get
        failed = original.response()
        failed.status_code = 500
        failed.raise_for_status.side_effect = requests.HTTPError(response=failed)
        self.post.return_value = failed
        config = {'video_gen': {'replicate': {'prediction_route': 'official_name'}}}
        with patch('hermes_cli.config.load_config_readonly', return_value=config):
            self.assert_failure(lambda: self.video.generate('contract'), 'api_error')
        self.post.assert_called_once()
        self.assertEqual(self.post.call_args.args[0], backend._API)

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
