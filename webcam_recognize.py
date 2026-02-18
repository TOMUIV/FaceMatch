import eventlet
eventlet.monkey_patch()

import os
import cv2
import numpy as np
import pickle
import base64
import logging
import ssl
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# ============================ 配置参数 ============================
class WebConfig:
    # 模型配置
    MODEL_NAME = 'buffalo_l'
    DET_SIZE = (480, 480)  # 平衡检测分辨率和速度
    
    # 识别配置
    SIMILARITY_THRESHOLD = 0.6
    MIN_FACE_CONFIDENCE = 0.3
    
    # 显示配置
    BOX_COLOR_KNOWN = (0, 255, 0)    # 绿色 - 已知人脸
    BOX_COLOR_UNKNOWN = (255, 0, 0)  # 蓝色 - 未知人脸
    TEXT_COLOR = (255, 255, 255)     # 白色文字
    
    # 网络配置
    HOST = '::'  # 监听所有接口
    PORT = 6001  # 人脸识别服务端口
    DEBUG = False
    
    # 截图配置
    SCREENSHOT_FOLDER = './screenshots'
    
    # ONNX Runtime配置
    FORCE_CPU = False
    
    # SSL证书配置
    SSL_CERTIFICATE = ''
    SSL_PRIVATE_KEY = ''
    
    # 性能优化配置
    MAX_PROCESSING_FPS = 10  # 最大处理帧率（提高以允许更多帧通过）
    SKIP_FRAMES_WHEN_BUSY = True  # 忙时跳过帧
    FRAME_QUEUE_MAX_SIZE = 2  # 帧队列最大大小，超过则丢弃旧帧

# ============================ 人脸识别器 ============================
class WebFaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.face_database = {}
        self.is_processing = False
        self.screenshot_count = 0
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
    
    def load_database(self, database_path='face_database.pkl'):
        """加载人脸数据库"""
        self.logger.info(f"加载人脸数据库: {database_path}")
        
        if not Path(database_path).exists():
            self.logger.error(f"数据库文件不存在: {database_path}")
            return False
        
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
            
            self.face_database = database.get('face_database', {})
            self.logger.info(f"✓ 数据库加载成功，包含 {len(self.face_database)} 个已知人物")
            
            for person_name, data in self.face_database.items():
                self.logger.info(f"  - {person_name}: {data.get('sample_count', 0)} 张样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据库加载失败: {e}")
            return False
    
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
            # 解码base64图像数据
            if ',' not in image_data:
                return None, []
            
            image_bytes = base64.b64decode(image_data.split(',')[1])
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}_{client_id}_{self.screenshot_count:04d}.jpg"
            filepath = Path(self.config.SCREENSHOT_FOLDER) / filename
            
            # 解码并保存图像
            image_bytes = base64.b64decode(image_data.split(',')[1])
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
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
app = Flask(__name__)
app.config['SECRET_KEY'] = 'web_face_recognition_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 全局识别器实例
recognizer = None

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/stats')
def get_stats():
    """获取统计信息"""
    if recognizer:
        return jsonify(recognizer.get_stats())
    return jsonify({'error': '识别器未初始化'})

@socketio.on('connect')
def handle_connect():
    """客户端连接事件"""
    if recognizer:
        recognizer.processing_stats['active_clients'] += 1
        client_id = request.sid
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
    
    config = WebConfig()
    
    try:
        # 初始化识别器
        recognizer = WebFaceRecognizer(config)
        
        # 加载人脸数据库
        if not recognizer.load_database('face_database.pkl'):
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
            print("⚠ 未找到SSL证书，将使用HTTP协议")
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