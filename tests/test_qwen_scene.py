"""Scene API and local model protocol; no real model or camera required."""
import io
import json
import threading
import unittest
import urllib.error
from unittest.mock import patch

from PIL import Image
from web.qwen_scene import QwenSceneService, MAX_IMAGE_BYTES, scene_text
from web.server import create_app
from web.services import ServiceError


def image_bytes(fmt='PNG', size=(30, 20)):
    out = io.BytesIO()
    Image.new('RGB', size, 'blue').save(out, fmt)
    return out.getvalue()


GOOD = {'done': True, 'message': {'content': json.dumps({'objects': [
    {'category': 'bottle', 'color': 'blue', 'position': 'center'}]})}}


class Reply(io.BytesIO):
    def __init__(self, body):
        super().__init__(json.dumps(body).encode())


class SceneTests(unittest.TestCase):
    def service(self, replies=None):
        replies = iter(replies if replies is not None else [{'capabilities': ['vision']}, GOOD])
        self.calls = []
        def opener(req, timeout):
            self.calls.append((req.full_url, json.loads(req.data), timeout))
            value = next(replies)
            if isinstance(value, Exception):
                raise value
            return Reply(value)
        return QwenSceneService(opener=opener)

    def post(self, service, data=None, mime='image/png'):
        client = create_app(scene_service=service).test_client()
        return client.post('/api/scene/describe', data={
            'image': (io.BytesIO(image_bytes() if data is None else data), 'frame', mime)})

    def test_success_png_and_jpeg(self):
        for fmt, mime in [('PNG', 'image/png'), ('JPEG', 'image/jpeg')]:
            response = self.post(self.service(), image_bytes(fmt), mime)
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertIn('blue bottle', body['description'])
            self.assertEqual(body['model'], 'qwen2.5vl:3b')
            self.assertFalse(body['product_identity_verified'])
            self.assertGreaterEqual(body['processing_time_ms'], 0)
            self.assertEqual(response.headers['Cache-Control'], 'no-store')
            url, payload, timeout = self.calls[-1]
            self.assertEqual(url, 'http://127.0.0.1:11434/api/chat')
            self.assertEqual(len(payload['messages'][0]['images']), 1)
            self.assertFalse(payload['stream'])
            self.assertLessEqual(timeout, 60)

    def test_missing_and_empty(self):
        client = create_app(scene_service=self.service()).test_client()
        self.assertEqual(client.post('/api/scene/describe').get_json()['error']['code'], 'missing_image')
        self.assertEqual(self.post(self.service(), b'').status_code, 400)

    def test_invalid_and_mismatched_image(self):
        for data, mime in [(b'not an image', 'image/png'), (image_bytes(), 'image/jpeg'),
                           (image_bytes('GIF'), 'image/gif')]:
            self.assertEqual(self.post(self.service(), data, mime).status_code, 400)
            self.assertEqual(self.calls, [])

    def test_byte_and_pixel_limits(self):
        self.assertEqual(self.post(self.service(), b'x' * (MAX_IMAGE_BYTES + 1)).status_code, 413)
        self.assertEqual(self.post(self.service(), image_bytes(size=(4000, 3100))).status_code, 413)
        self.assertEqual(self.calls, [])

    def test_resizes_and_removes_metadata(self):
        self.post(self.service(), image_bytes(size=(2000, 1000)))
        import base64
        data = base64.b64decode(self.calls[-1][1]['messages'][0]['images'][0])
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.size, (1024, 512))
            self.assertFalse(image.getexif())

    def test_unavailable_timeout_and_invalid(self):
        cases = [([urllib.error.URLError('refused')], 503, 'model_unavailable'),
                 ([TimeoutError()], 504, 'model_timeout'),
                 ([{'capabilities': ['vision']}, TimeoutError()], 504, 'model_timeout'),
                 ([{'capabilities': ['completion']}], 503, 'model_not_vision'),
                 ([{'capabilities': ['vision'], 'remote_host': 'https://ollama.com'}], 503, 'model_not_local'),
                 ([{'capabilities': ['vision']}, {}], 502, 'invalid_model_response'),
                 ([{'capabilities': ['vision']}, {'done': True, 'message': {'content': 'hello'}}], 502, 'invalid_model_response'),
                 ([[]], 502, 'invalid_model_response'),
                 ([{'capabilities': None}], 502, 'invalid_model_response'),
                 ([{'capabilities': ['vision']}, dict(GOOD, done_reason='length')], 502, 'invalid_model_response')]
        for replies, status, code in cases:
            response = self.post(self.service(replies))
            self.assertEqual((response.status_code, response.get_json()['error']['code']), (status, code))

    def test_unsafe_or_malformed_output_is_not_displayed(self):
        for content in ['{"objects":[{"category":"ibuprofen","color":"blue","position":"center"}]}',
                        '{"objects":[],"advice":"take 2 tablets"}', 'null', '{"objects":[null]}']:
            with self.assertRaises(ServiceError):
                scene_text(content)

    def test_lock_released_after_failure(self):
        service = self.service([TimeoutError(), {'capabilities': ['vision']}, GOOD])
        self.assertEqual(self.post(service).status_code, 504)
        self.assertEqual(self.post(service).status_code, 200)

    def test_concurrent_requests_rejected_without_queue(self):
        entered, release = threading.Event(), threading.Event()
        service = self.service()
        original = service._post
        def blocking(path, payload, deadline):
            if path == 'chat':
                entered.set()
                release.wait(5)
            return original(path, payload, deadline)
        service._post = blocking
        results = []
        worker = threading.Thread(target=lambda: results.append(self.post(service).status_code))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            response = self.post(service)
            self.assertEqual((response.status_code, response.get_json()['error']['code']), (409, 'scene_busy'))
            # Other existing endpoints remain accessible while Qwen is occupied.
            client = create_app(scene_service=service).test_client()
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertEqual(client.post('/api/detect').get_json()['error']['code'], 'missing_image')
        finally:
            release.set()
            worker.join(5)
        self.assertEqual(results, [200])

    def test_upload_stream_stays_in_memory(self):
        from web.server import CameraRequest
        with create_app().test_request_context('/api/scene/describe'):
            request = CameraRequest.from_values('/api/scene/describe')
            stream = request._get_file_stream(900000, 'image/jpeg', 'scene.jpg')
            self.assertIsInstance(stream, io.BytesIO)
            stream.close()
            request.close()

    def test_model_is_configurable(self):
        with patch.dict('os.environ', {'QWEN_VISION_MODEL': 'qwen2.5vl:3b'}):
            self.assertEqual(QwenSceneService().model, 'qwen2.5vl:3b')
