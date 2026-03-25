import eventlet
eventlet.monkey_patch()

import binascii
import os
import cv2
import numpy as np
import base64
import logging
import secrets
import ssl
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from safe_storage import (
    DEFAULT_DATABASE_PATH,
    load_face_database,
    safe_child_path,
    sanitize_filename_component,
)

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数") from exc


def _is_loopback_host(host):
    normalized = (host or '').strip().strip('[]').lower()
    return normalized in {'127.0.0.1', 'localhost', '::1'}


def _format_origin_host(host):
    normalized = (host or '').strip()
    if not normalized:
        return normalized
    if ':' in normalized and not normalized.startswith('['):
        return f'[{normalized}]'
    return normalized


def _default_allowed_origins(host, port):
    origins = {
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        f"http://[::1]:{port}",
        f"https://127.0.0.1:{port}",
        f"https://localhost:{port}",
        f"https://[::1]:{port}",
    }
    normalized_host = (host or '').strip()
    if normalized_host and normalized_host not in {'0.0.0.0', '::'}:
        origin_host = _format_origin_host(normalized_host)
        origins.add(f"http://{origin_host}:{port}")
        origins.add(f"https://{origin_host}:{port}")
    return sorted(origins)


def _load_allowed_origins(host, port):
    origins_value = os.environ.get('FACE_MATCH_ALLOWED_ORIGINS')
    if not origins_value:
        return _default_allowed_origins(host, port)

    origins = [origin.strip() for origin in origins_value.split(',') if origin.strip()]
    if any(origin == '*' for origin in origins):
        raise ValueError("FACE_MATCH_ALLOWED_ORIGINS 不能设置为 '*'")
    return origins


def _constant_time_token_match(provided_token, expected_token):
    if not provided_token or not expected_token:
        return False
    return secrets.compare_digest(str(provided_token), str(expected_token))


def _extract_socket_token(auth_payload=None):
    if isinstance(auth_payload, dict):
        token = auth_payload.get('token')
        if token:
            return token

    args = getattr(request, 'args', None)
    if args and hasattr(args, 'get'):
        token = args.get('token')
        if token:
            return token

    headers = getattr(request, 'headers', None)
    if headers and hasattr(headers, 'get'):
        return headers.get('X-FaceMatch-Token')

    return None


def _extract_http_token():
    headers = getattr(request, 'headers', None)
    if headers and hasattr(headers, 'get'):
        token = headers.get('X-FaceMatch-Token')
        if token:
            return token

    args = getattr(request, 'args', None)
    if args and hasattr(args, 'get'):
        return args.get('token')

    return None


def _is_authorized_socket_request(auth_payload=None):
    token = _extract_socket_token(auth_payload)
    if _constant_time_token_match(token, app.config.get('FACE_MATCH_AUTH_TOKEN')):
        return True

    client_id = getattr(request, 'sid', None)
    return client_id in authorized_clients


def _is_authorized_http_request():
    return _constant_time_token_match(
        _extract_http_token(),
        app.config.get('FACE_MATCH_AUTH_TOKEN')
    )


def _security_logger():
    if recognizer and hasattr(recognizer, 'logger'):
        return recognizer.logger
    return logging.getLogger(__name__)


# ============================ 配置参数 ============================
class WebConfig:
    def __init__(self):
        # 模型配置
        self.MODEL_NAME = 'buffalo_l'
        self.DET_SIZE = (480, 480)  # 平衡检测分辨率和速度
        
        # 识别配置
        self.SIMILARITY_THRESHOLD = 0.6
        self.MIN_FACE_CONFIDENCE = 0.3
        
        # 显示配置
        self.BOX_COLOR_KNOWN = (0, 255, 0)    # 绿色 - 已知人脸
        self.BOX_COLOR_UNKNOWN = (255, 0, 0)  # 蓝色 - 未知人脸
        self.TEXT_COLOR = (255, 255, 255)     # 白色文字
        
        # 网络配置
        self.HOST = os.environ.get('FACE_MATCH_HOST', '127.0.0.1')
        self.PORT = _env_int('FACE_MATCH_PORT', 6001)
        self.DEBUG = _env_flag('FACE_MATCH_DEBUG', False)
        self.ALLOW_INSECURE_HTTP = _env_flag('FACE_MATCH_ALLOW_INSECURE_HTTP', False)
        self.ALLOWED_ORIGINS = _load_allowed_origins(self.HOST, self.PORT)
        
        # 截图配置
        self.SCREENSHOT_FOLDER = os.environ.get('FACE_MATCH_SCREENSHOT_FOLDER', './screenshots')
        self.MAX_FRAME_BYTES = max(1024, _env_int('FACE_MATCH_MAX_FRAME_BYTES', 2 * 1024 * 1024))
        self.MAX_SCREENSHOT_BYTES = max(1024, _env_int('FACE_MATCH_MAX_SCREENSHOT_BYTES', 2 * 1024 * 1024))
        self.MAX_SCREENSHOTS_PER_MINUTE = max(1, _env_int('FACE_MATCH_MAX_SCREENSHOTS_PER_MINUTE', 12))
        self.MAX_SCREENSHOT_FILES = max(1, _env_int('FACE_MATCH_MAX_SCREENSHOT_FILES', 200))
        
        # ONNX Runtime配置
        self.FORCE_CPU = False
        
        # SSL证书配置
        self.SSL_CERTIFICATE = os.environ.get('FACE_MATCH_SSL_CERTIFICATE', '')
        self.SSL_PRIVATE_KEY = os.environ.get('FACE_MATCH_SSL_PRIVATE_KEY', '')
        
        # 认证配置
        self.SECRET_KEY = os.environ.get('FACE_MATCH_SECRET_KEY') or secrets.token_hex(32)
        self.AUTH_TOKEN = os.environ.get('FACE_MATCH_AUTH_TOKEN') or secrets.token_urlsafe(32)
        
        # 性能优化配置
        self.MAX_PROCESSING_FPS = 10  # 最大处理帧率（提高以允许更多帧通过）
        self.SKIP_FRAMES_WHEN_BUSY = True  # 忙时跳过帧
        self.FRAME_QUEUE_MAX_SIZE = 2  # 帧队列最大大小，超过则丢弃旧帧

# ============================ 人脸识别器 ============================
class WebFaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.face_database = {}
        self.is_processing = False
        self.screenshot_count = 0
        self.screenshot_history = {}
        self.processing_stats = {
            'total_frames': 0,
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'start_time': datetime.now(),
            'active_clients': 0
        }
        
        # 性能优化相关
        self.is_currently_processing = False  # 是否正在处理中
        self.latest_frame = None  # 最新帧数据
        
        self._setup_logging()
        self._setup_screenshot_folder()
        self._initialize_model()
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建日志目录
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True, parents=True)
        log_file = log_dir / 'web_face_recognition.log'
        
        # 完全抑制insightface和onnxruntime的日志
        logging.getLogger('insightface').setLevel(logging.ERROR)
        logging.getLogger('onnxruntime').setLevel(logging.ERROR)
        logging.getLogger('engine').setLevel(logging.ERROR)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== Web人脸识别系统启动 ===")
        self.logger.info(f"日志文件: {log_file.absolute()}")
    
    def _setup_screenshot_folder(self):
        """创建截图文件夹"""
        screenshot_path = Path(self.config.SCREENSHOT_FOLDER)
        screenshot_path.mkdir(exist_ok=True)
        self.logger.info(f"截图文件夹: {screenshot_path.absolute()}")
    
    def _initialize_model(self):
        """初始化人脸识别模型 - 完全抑制日志输出"""
        self.logger.info("正在初始化人脸识别模型...")
        
        try:
            # 重定向所有标准输出到null
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            
            from insightface.app import FaceAnalysis
            
            # 配置ONNX Runtime providers
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers and not self.config.FORCE_CPU:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    self.logger.info("✓ 使用GPU加速")
                else:
                    providers = ['CPUExecutionProvider']
                    self.logger.info("ℹ 使用CPU模式")
            except ImportError:
                providers = ['CPUExecutionProvider']
                self.logger.info("ℹ 使用默认CPU模式")
            
            # 完全静默初始化模型
            self.app = FaceAnalysis(
                name=self.config.MODEL_NAME,
                providers=providers
            )
            
            self.app.prepare(
                ctx_id=0 if providers[0] == 'CUDAExecutionProvider' else -1,
                det_size=self.config.DET_SIZE
            )
            
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            self.logger.info("✓ 模型初始化完成")
            
        except Exception as e:
            # 确保恢复标准输出
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            self.logger.error(f"模型初始化失败: {e}")
            raise
    
    def load_database(self, database_path=DEFAULT_DATABASE_PATH):
        """加载人脸数据库"""
        self.logger.info(f"加载人脸数据库: {database_path}")
        
        try:
            database = load_face_database(database_path)
            self.face_database = database.get('face_database', {})
            self.logger.info(f"✓ 数据库加载成功，包含 {len(self.face_database)} 个已知人物")
            if database.get('migrated_from_legacy'):
                self.logger.info(f"✓ 已将旧版数据库迁移到安全格式: {database['database_path']}")
            
            for person_name, data in self.face_database.items():
                self.logger.info(f"  - {person_name}: {data.get('sample_count', 0)} 张样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据库加载失败: {e}")
            return False

    def _decode_image_bytes(self, image_data, max_bytes):
        """解码并校验base64图像数据"""
        if not isinstance(image_data, str) or ',' not in image_data:
            raise ValueError("图像数据格式无效")

        payload = image_data.split(',', 1)[1]
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("图像数据解码失败") from exc

        if not image_bytes:
            raise ValueError("图像数据为空")

        if len(image_bytes) > max_bytes:
            raise ValueError(f"图像数据超过大小限制: {max_bytes} bytes")

        return image_bytes

    def _check_screenshot_rate_limit(self, client_id):
        """限制单个客户端的截图频率"""
        now = time.monotonic()
        history = self.screenshot_history.setdefault(client_id, deque())
        window_start = now - 60

        while history and history[0] < window_start:
            history.popleft()

        if len(history) >= self.config.MAX_SCREENSHOTS_PER_MINUTE:
            raise ValueError("截图请求过于频繁，请稍后再试")

        history.append(now)

    def _prune_screenshot_folder(self):
        """删除旧截图，避免截图目录无限增长"""
        screenshot_path = Path(self.config.SCREENSHOT_FOLDER)
        screenshot_files = sorted(
            screenshot_path.glob('screenshot_*.jpg'),
            key=lambda path: path.stat().st_mtime
        )
        excess_count = len(screenshot_files) - self.config.MAX_SCREENSHOT_FILES + 1

        for old_file in screenshot_files[:max(0, excess_count)]:
            try:
                old_file.unlink()
            except OSError as exc:
                self.logger.warning(f"删除旧截图失败 {old_file}: {exc}")
    
    def _calculate_similarity(self, embedding1, embedding2):
        """计算余弦相似度"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2)
    
    def _recognize_face(self, face_embedding):
        """识别人脸"""
        best_similarity = 0
        best_person = "未知"
        
        for person_name, person_data in self.face_database.items():
            similarity = self._calculate_similarity(face_embedding, person_data['embedding'])
            
            if similarity > best_similarity and similarity >= self.config.SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_person = person_name
        
        return best_person, best_similarity
    
    def process_frame(self, image_data):
        """处理单帧图像"""
        try:
            image_bytes = self._decode_image_bytes(image_data, self.config.MAX_FRAME_BYTES)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, []
            
            # 更新统计信息
            self.processing_stats['total_frames'] += 1
            
            # 检测人脸
            faces = self.app.get(image)
            recognition_results = []
            
            for face in faces:
                confidence = getattr(face, 'det_score', getattr(face, 'det', 0))
                
                if confidence < self.config.MIN_FACE_CONFIDENCE:
                    continue
                
                # 识别人脸
                person_name, similarity = self._recognize_face(face.embedding)
                
                # 准备结果数据
                result = {
                    'bbox': face.bbox.tolist(),
                    'person_name': person_name,
                    'similarity': float(similarity),
                    'confidence': float(confidence)
                }
                recognition_results.append(result)
                
                # 更新统计信息
                self.processing_stats['total_faces'] += 1
                if person_name != "未知":
                    self.processing_stats['recognized_faces'] += 1
                else:
                    self.processing_stats['unknown_faces'] += 1
            
            return image, recognition_results
            
        except Exception as e:
            self.logger.error(f"处理帧失败: {e}")
            return None, []
    
    def capture_screenshot(self, image_data, client_id):
        """保存截图"""
        try:
            self._check_screenshot_rate_limit(client_id)
            self._prune_screenshot_folder()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_client_id = sanitize_filename_component(client_id, fallback="client")
            filename = f"screenshot_{timestamp}_{safe_client_id}_{self.screenshot_count:04d}.jpg"
            filepath = safe_child_path(self.config.SCREENSHOT_FOLDER, filename)
            
            # 解码并重编码图像，避免持久化任意字节流
            image_bytes = self._decode_image_bytes(image_data, self.config.MAX_SCREENSHOT_BYTES)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("无法解码截图图像")

            success, encoded_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                raise ValueError("截图重编码失败")

            encoded_bytes = encoded_image.tobytes() if hasattr(encoded_image, 'tobytes') else bytes(encoded_image)
            with open(filepath, 'wb') as f:
                f.write(encoded_bytes)
            
            self.screenshot_count += 1
            self.logger.info(f"截图已保存: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"截图失败: {e}")
            return None
    
    def get_stats(self):
        """获取统计信息"""
        stats = self.processing_stats.copy()
        elapsed_time = (datetime.now() - stats['start_time']).total_seconds()
        stats['elapsed_time'] = f"{elapsed_time:.1f}秒"
        stats['fps'] = stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
        return stats

# ============================ Flask应用 ============================
runtime_config = WebConfig()
app = Flask(__name__)
app.config['SECRET_KEY'] = runtime_config.SECRET_KEY
app.config['FACE_MATCH_AUTH_TOKEN'] = runtime_config.AUTH_TOKEN
socketio = SocketIO(app, cors_allowed_origins=runtime_config.ALLOWED_ORIGINS, async_mode='eventlet')

# 全局识别器实例
recognizer = None
authorized_clients = set()

@app.route('/')
def index():
    """主页面"""
    return render_template(
        'index.html',
        socket_auth_token=app.config['FACE_MATCH_AUTH_TOKEN'],
        cors_allowed_origins=runtime_config.ALLOWED_ORIGINS,
    )

@app.route('/stats')
def get_stats():
    """获取统计信息"""
    if not _is_authorized_http_request():
        return jsonify({'error': '未授权'}), 403
    if recognizer:
        return jsonify(recognizer.get_stats())
    return jsonify({'error': '识别器未初始化'})

@socketio.on('connect')
def handle_connect(auth=None):
    """客户端连接事件"""
    if not _is_authorized_socket_request(auth):
        _security_logger().warning(
            f"拒绝未授权的Socket连接: sid={getattr(request, 'sid', 'unknown')}"
        )
        return False

    if recognizer:
        recognizer.processing_stats['active_clients'] += 1
        client_id = request.sid
        authorized_clients.add(client_id)
        recognizer.logger.info(f"客户端连接: {client_id}, 活跃客户端: {recognizer.processing_stats['active_clients']}")
        emit('connected', {
            'message': '连接成功',
            'client_id': client_id,
            'server_time': datetime.now().isoformat()
        })
    else:
        emit('error', {'message': '识别器未初始化'})

@socketio.on('disconnect')
def handle_disconnect(*args):
    """客户端断开连接事件"""
    try:
        client_id = getattr(request, 'sid', None)
        if client_id in authorized_clients:
            authorized_clients.discard(client_id)
        if recognizer:
            recognizer.processing_stats['active_clients'] = max(0, recognizer.processing_stats['active_clients'] - 1)
    except Exception as e:
        pass


@socketio.on_error_default
def default_error_handler(e):
    """处理WebSocket错误"""
    if recognizer:
        recognizer.logger.error(f"WebSocket错误: {str(e)}")
    emit('error', {'message': f'服务器错误: {str(e)}'})

@socketio.on('frame')
def handle_frame(data):
    """处理客户端发送的视频帧 - 使用线程池异步处理"""
    try:
        if not recognizer:
            return

        if not _is_authorized_socket_request():
            emit('error', {'message': '未授权请求'})
            return
            
        client_id = request.sid
        
        # 更新最新帧数据（始终保存最新帧）
        recognizer.latest_frame = {
            'data': data,
            'client_id': client_id,
            'timestamp': data.get('timestamp', 0)
        }
        
        # 如果已经在处理中，直接返回（不处理此帧，让后台线程处理最新帧）
        if recognizer.is_currently_processing:
            return
        
        # 标记为正在处理
        recognizer.is_currently_processing = True
        
        # 使用线程池异步处理帧，避免阻塞WebSocket
        def process_worker():
            try:
                while recognizer.latest_frame:
                    frame_data = recognizer.latest_frame
                    recognizer.latest_frame = None
                    
                    _process_frame_internal(frame_data['data'], frame_data['client_id'])
                    
                    # 短暂休眠，让其他事件有机会处理
                    socketio.sleep(0.001)
            finally:
                recognizer.is_currently_processing = False
        
        # 启动异步处理
        socketio.start_background_task(process_worker)
            
    except Exception as e:
        if recognizer:
            recognizer.logger.error(f"处理帧数据失败: {e}")
        recognizer.is_currently_processing = False

def _process_frame_internal(data, client_id):
    """内部帧处理函数"""
    try:
        # 获取前端发送的时间戳（用于计算延迟）
        client_timestamp = data.get('timestamp', 0)
        image_data = data.get('image_data', '')
        
        if not image_data:
            return
        
        # 处理图像帧
        image, results = recognizer.process_frame(image_data)
        
        if image is not None:
            # 发送识别结果回客户端（使用socketio.emit避免请求上下文问题）
            response_data = {
                'results': results,
                'client_timestamp': client_timestamp,
                'server_timestamp': int(datetime.now().timestamp() * 1000),
                'timestamp': datetime.now().isoformat()
            }
            socketio.emit('recognition_result', response_data, room=client_id)
    except Exception as e:
        if recognizer:
            recognizer.logger.error(f"内部处理帧失败: {e}")

@socketio.on('screenshot')
def handle_screenshot(data):
    """处理截图请求"""
    try:
        if not recognizer:
            emit('screenshot_error', {'message': '识别器未初始化'})
            return

        if not _is_authorized_socket_request():
            emit('screenshot_error', {'message': '未授权请求'})
            return
            
        client_id = request.sid
        screenshot_path = recognizer.capture_screenshot(data['image_data'], client_id)
        
        if screenshot_path:
            # 返回相对路径供前端使用
            relative_path = screenshot_path.replace('\\', '/')
            emit('screenshot_saved', {
                'path': screenshot_path,
                'relative_path': relative_path,
                'filename': Path(screenshot_path).name
            })
            recognizer.logger.info(f"客户端 {client_id}: 截图已保存")
        else:
            emit('screenshot_error', {'message': '截图保存失败'})
            
    except Exception as e:
        if recognizer:
            recognizer.logger.error(f"截图处理失败: {e}")
        emit('screenshot_error', {'message': str(e)})

def create_ssl_context(certfile, keyfile):
    """创建SSL上下文"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context

def main():
    """主函数"""
    global recognizer
    
    config = runtime_config
    
    try:
        # 初始化识别器
        recognizer = WebFaceRecognizer(config)
        recognizer.logger.info(f"允许的Socket来源: {', '.join(config.ALLOWED_ORIGINS)}")
        
        # 加载人脸数据库
        if not recognizer.load_database(DEFAULT_DATABASE_PATH):
            print("❌ 数据库加载失败，请先运行训练程序")
            return
        
        # 检查SSL证书是否存在
        use_ssl = os.path.exists(config.SSL_CERTIFICATE) and os.path.exists(config.SSL_PRIVATE_KEY)
        
        if use_ssl:
            print(f"✓ 找到SSL证书: {config.SSL_CERTIFICATE}")
            print(f"✓ 找到SSL私钥: {config.SSL_PRIVATE_KEY}")
            
            # 修改：让Flask-SocketIO自动处理SSL
            socketio.run(
                app,
                host=config.HOST,  # 使用 '::'
                port=config.PORT,
                certfile=config.SSL_CERTIFICATE,
                keyfile=config.SSL_PRIVATE_KEY,
                debug=config.DEBUG,
                log_output=False
            )
        else:
            if not (_is_loopback_host(config.HOST) or config.ALLOW_INSECURE_HTTP):
                print("❌ 未找到SSL证书，且当前监听地址不是本地回环地址。")
                print("请配置 HTTPS 证书，或将 FACE_MATCH_HOST 设为 127.0.0.1/localhost，")
                print("如确需在非本地接口上使用 HTTP，请显式设置 FACE_MATCH_ALLOW_INSECURE_HTTP=1。")
                return

            print("⚠ 未找到SSL证书，将使用受限HTTP模式")
            print(f"监听地址: {config.HOST}")
            print(f"证书路径: {config.SSL_CERTIFICATE}")
            print(f"私钥路径: {config.SSL_PRIVATE_KEY}")
            
            # HTTP启动
            socketio.run(
                app,
                host=config.HOST,  # 使用 '::'
                port=config.PORT,
                debug=config.DEBUG,
                log_output=False
            )
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
