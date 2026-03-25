import base64
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def install_webcam_stubs():
    class FakeArray:
        def __init__(self, data):
            self._data = list(data)

        def tobytes(self):
            return bytes(self._data)

        def tolist(self):
            return list(self._data)

        def __iter__(self):
            return iter(self._data)

        def __len__(self):
            return len(self._data)

        def __truediv__(self, scalar):
            return FakeArray([value / scalar for value in self._data])

    numpy_module = types.ModuleType('numpy')
    numpy_module.uint8 = int
    numpy_module.float32 = float
    numpy_module.number = (int, float)
    numpy_module.frombuffer = lambda payload, dtype=None: FakeArray(payload)
    numpy_module.dot = lambda left, right: sum(a * b for a, b in zip(list(left), list(right)))
    numpy_module.linalg = types.SimpleNamespace(
        norm=lambda value: max(1.0, sum(float(item) * float(item) for item in value) ** 0.5)
    )
    sys.modules['numpy'] = numpy_module

    cv2_module = types.ModuleType('cv2')
    cv2_module.IMREAD_COLOR = 1
    cv2_module.IMWRITE_JPEG_QUALITY = 1

    class FakeImage:
        def copy(self):
            return self

    class FakeEncodedImage:
        def __init__(self, payload):
            self.payload = payload

        def tobytes(self):
            return self.payload

    cv2_module.imdecode = lambda image_array, flag: FakeImage() if len(image_array) else None
    cv2_module.imencode = lambda ext, image, params=None: (True, FakeEncodedImage(b'jpeg-bytes'))
    sys.modules['cv2'] = cv2_module

    eventlet_module = types.ModuleType('eventlet')
    eventlet_module.monkey_patch = lambda: None
    sys.modules['eventlet'] = eventlet_module

    flask_module = types.ModuleType('flask')

    class FakeFlask:
        def __init__(self, name):
            self.name = name
            self.config = {}
            self.routes = {}

        def route(self, path, **kwargs):
            def decorator(fn):
                self.routes[path] = fn
                return fn
            return decorator

    class FakeRequest:
        def __init__(self):
            self.sid = 'test-client'
            self.headers = {}
            self.args = {}

    flask_module.Flask = FakeFlask
    flask_module.request = FakeRequest()
    flask_module.render_template = lambda template_name, **kwargs: {'template': template_name, **kwargs}
    flask_module.jsonify = lambda value=None, **kwargs: value if value is not None else kwargs
    sys.modules['flask'] = flask_module

    socketio_module = types.ModuleType('flask_socketio')

    class FakeSocketIO:
        def __init__(self, app, cors_allowed_origins=None, async_mode=None):
            self.app = app
            self.cors_allowed_origins = cors_allowed_origins
            self.async_mode = async_mode
            self.handlers = {}

        def on(self, event_name):
            def decorator(fn):
                self.handlers[event_name] = fn
                return fn
            return decorator

        def on_error_default(self, fn):
            self.handlers['__default_error__'] = fn
            return fn

        def emit(self, event_name, data=None, room=None):
            return None

        def sleep(self, seconds):
            return None

        def start_background_task(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

        def run(self, app, **kwargs):
            return None

    socketio_module.SocketIO = FakeSocketIO
    socketio_module.emit = lambda *args, **kwargs: None
    sys.modules['flask_socketio'] = socketio_module

    insightface_module = types.ModuleType('insightface')
    insightface_app_module = types.ModuleType('insightface.app')

    class FakeFace:
        def __init__(self):
            self.det_score = 0.99
            self.bbox = FakeArray([10, 20, 30, 40])
            self.embedding = FakeArray([0.6, 0.8])

    class FaceAnalysis:
        def __init__(self, *args, **kwargs):
            return None

        def prepare(self, *args, **kwargs):
            return None

        def get(self, image):
            return [FakeFace()]

    insightface_app_module.FaceAnalysis = FaceAnalysis
    insightface_module.app = insightface_app_module
    sys.modules['insightface'] = insightface_module
    sys.modules['insightface.app'] = insightface_app_module


class DummyLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class DummyRecognizer:
    def __init__(self):
        self.processing_stats = {'active_clients': 0}
        self.logger = DummyLogger()

    def get_stats(self):
        return {'active_clients': self.processing_stats['active_clients']}


class WebcamSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_webcam_stubs()
        sys.modules.pop('webcam_recognize', None)
        cls.module = importlib.import_module('webcam_recognize')

    def setUp(self):
        self.module.authorized_clients.clear()
        self.module.recognizer = DummyRecognizer()
        self.module.request.sid = 'test-client'
        self.module.request.headers = {}
        self.module.request.args = {}

    def test_secret_key_and_cors_are_not_insecure_defaults(self):
        self.assertNotEqual(self.module.app.config['SECRET_KEY'], 'web_face_recognition_secret')
        self.assertNotEqual(self.module.socketio.cors_allowed_origins, '*')
        self.assertEqual(self.module.runtime_config.HOST, '127.0.0.1')

    def test_connect_rejects_missing_token(self):
        result = self.module.handle_connect(auth={})
        self.assertFalse(result)
        self.assertEqual(self.module.recognizer.processing_stats['active_clients'], 0)
        self.assertNotIn('test-client', self.module.authorized_clients)

    def test_connect_accepts_valid_token_and_tracks_client(self):
        token = self.module.app.config['FACE_MATCH_AUTH_TOKEN']
        result = self.module.handle_connect(auth={'token': token})
        self.assertIsNone(result)
        self.assertEqual(self.module.recognizer.processing_stats['active_clients'], 1)
        self.assertIn('test-client', self.module.authorized_clients)

    def test_stats_endpoint_requires_auth_token(self):
        response = self.module.get_stats()
        self.assertEqual(response[1], 403)
        self.assertEqual(response[0]['error'], '未授权')

        self.module.request.headers = {'X-FaceMatch-Token': self.module.app.config['FACE_MATCH_AUTH_TOKEN']}
        stats = self.module.get_stats()
        self.assertEqual(stats['active_clients'], 0)

    def test_capture_screenshot_rejects_oversized_payloads_and_rate_limits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.module.WebConfig()
            config.SCREENSHOT_FOLDER = temp_dir
            config.MAX_SCREENSHOT_BYTES = 8
            config.MAX_SCREENSHOTS_PER_MINUTE = 1
            recognizer = self.module.WebFaceRecognizer(config)

            oversized_payload = 'data:image/jpeg;base64,' + base64.b64encode(b'0123456789').decode('ascii')
            self.assertIsNone(recognizer.capture_screenshot(oversized_payload, 'oversize-client'))

            small_payload = 'data:image/jpeg;base64,' + base64.b64encode(b'1234').decode('ascii')
            screenshot_path = recognizer.capture_screenshot(small_payload, '../client:id')
            self.assertIsNotNone(screenshot_path)
            self.assertTrue(Path(screenshot_path).exists())

            second_path = recognizer.capture_screenshot(small_payload, '../client:id')
            self.assertIsNone(second_path)


if __name__ == '__main__':
    unittest.main()
