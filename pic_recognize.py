import os
import cv2
import numpy as np
import pickle
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime
import sys
from io import StringIO

# ============================ 全局配置参数 ============================
class RecognizeConfig:
    # 模型配置
    MODEL_NAME = 'buffalo_l'
    DET_SIZE = (640, 640)
    
    # 识别配置
    SIMILARITY_THRESHOLD = 0.6  # 相似度阈值
    MIN_FACE_CONFIDENCE = 0.3   # 最小人脸检测置信度
    
    # 显示配置
    BOX_COLOR_KNOWN = (0, 255, 0)    # 绿色 - 已知人脸
    BOX_COLOR_UNKNOWN = (255, 0, 0)  # 蓝色 - 未知人脸
    TEXT_COLOR = (255, 255, 255)    # 白色文字
    TEXT_BG_COLOR = (0, 0, 0)       # 文字背景色
    BOX_THICKNESS = 3
    TEXT_THICKNESS = 2
    TEXT_SCALE = 1.0
    
    # 字体配置
    FONT_PATH = './msyh.ttc'  # 使用您提供的字体文件
    FONT_SIZE = 30  # 字体大小
    
    # 输出配置
    OUTPUT_FOLDER = './recognition_results'
    SAVE_IMAGES = True
    SHOW_IMAGES = False
    
    # ONNX Runtime GPU配置
    FORCE_CPU = False
    CUDA_DEVICE_ID = 0
    
    # 日志配置
    SUPPRESS_MODEL_LOGS = True

# ============================ 人脸识别器 ============================
class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.face_database = {}
        self.recognition_stats = {
            'total_images': 0,
            'total_faces': 0,
            'recognized_faces': 0,
            'unknown_faces': 0,
            'processing_time': 0,
            'start_time': datetime.now(),
            'gpu_available': False
        }
        self._setup_logging()
        self._check_onnxruntime_gpu()
        self._initialize_model()
        self._setup_output_folder()
    
    def _setup_logging(self):
        """设置日志"""
        if self.config.SUPPRESS_MODEL_LOGS:
            logging.getLogger('insightface').setLevel(logging.WARNING)
            logging.getLogger('onnxruntime').setLevel(logging.WARNING)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_recognition.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _check_onnxruntime_gpu(self):
        """检查ONNX Runtime GPU可用性"""
        try:
            import onnxruntime as ort
            # 获取可用的providers
            available_providers = ort.get_available_providers()
            self.logger.info(f"可用的ONNX Runtime providers: {available_providers}")
            
            if 'CUDAExecutionProvider' in available_providers and not self.config.FORCE_CPU:
                self.recognition_stats['gpu_available'] = True
                self.logger.info("✓ ONNX Runtime GPU可用")
            else:
                self.logger.info("ℹ 使用ONNX Runtime CPU进行推理")
                
        except ImportError:
            self.logger.info("ℹ 无法导入onnxruntime，使用默认配置")
    
    def _initialize_model(self):
        """初始化人脸分析模型 - 使用ONNX Runtime GPU加速"""
        self.logger.info("=== 初始化人脸识别模型 ===")
        
        try:
            # 设置GPU环境变量
            if not self.config.FORCE_CPU and self.recognition_stats['gpu_available']:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.config.CUDA_DEVICE_ID)
            
            # 临时重定向stdout来抑制模型加载日志
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            from insightface.app import FaceAnalysis
            
            # 配置providers - 优先使用GPU
            if not self.config.FORCE_CPU and self.recognition_stats['gpu_available']:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ctx_id = 0  # GPU设备ID
                self.logger.info("使用ONNX Runtime GPU加速")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1  # CPU
                self.logger.info("使用ONNX Runtime CPU")
            
            self.app = FaceAnalysis(
                name=self.config.MODEL_NAME,
                providers=providers
            )
            
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=self.config.DET_SIZE
            )
            
            sys.stdout = old_stdout
            self.logger.info("✓ 模型加载完成")
            
        except Exception as e:
            if 'old_stdout' in locals():
                sys.stdout = old_stdout
            self.logger.error(f"模型初始化失败: {e}")
            # 尝试回退到CPU
            self.logger.info("尝试使用CPU回退方案...")
            self._initialize_model_cpu_fallback()
    
    def _initialize_model_cpu_fallback(self):
        """CPU回退初始化"""
        try:
            from insightface.app import FaceAnalysis
            
            self.app = FaceAnalysis(
                name=self.config.MODEL_NAME,
                providers=['CPUExecutionProvider']
            )
            
            self.app.prepare(
                ctx_id=-1,
                det_size=self.config.DET_SIZE
            )
            self.logger.info("✓ 模型加载完成（CPU模式）")
        except Exception as e:
            self.logger.error(f"CPU回退也失败: {e}")
            raise
    
    def _setup_output_folder(self):
        """创建输出文件夹"""
        output_path = Path(self.config.OUTPUT_FOLDER)
        output_path.mkdir(exist_ok=True)
        self.logger.info(f"输出文件夹: {output_path.absolute()}")
    
    def load_database(self, database_path='face_database.pkl'):
        """加载人脸数据库"""
        self.logger.info(f"加载人脸数据库: {database_path}")
        
        if not Path(database_path).exists():
            self.logger.error(f"数据库文件不存在: {database_path}")
            return False
        
        try:
            with open(database_path, 'rb') as f:
                database = pickle.load(f)
            
            self.face_database = database['face_database']
            self.logger.info(f"✓ 数据库加载成功，包含 {len(self.face_database)} 个已知人物")
            
            # 打印已知人物列表
            for person_name, data in self.face_database.items():
                self.logger.info(f"  - {person_name}: {data['sample_count']} 张样本")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据库加载失败: {e}")
            return False
    
    def _calculate_similarity(self, embedding1, embedding2):
        """计算两个特征向量的余弦相似度"""
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(embedding1, embedding2)
        return similarity
    
    def _recognize_face(self, face_embedding):
        """识别人脸并返回最相似的人物名称和相似度"""
        best_similarity = 0
        best_person = "未知"
        
        for person_name, person_data in self.face_database.items():
            similarity = self._calculate_similarity(face_embedding, person_data['embedding'])
            
            if similarity > best_similarity and similarity >= self.config.SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_person = person_name
        
        return best_person, best_similarity
    
    def _draw_chinese_text(self, image, text, position, font_size, color, thickness):
        """使用指定字体绘制中文文本"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 检查字体文件是否存在
            if not Path(self.config.FONT_PATH).exists():
                self.logger.warning(f"字体文件不存在: {self.config.FONT_PATH}")
                return self._draw_opencv_text(image, text, position, font_size/30, color, thickness)
            
            # 将BGR图像转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # 加载字体
            font = ImageFont.truetype(self.config.FONT_PATH, font_size)
            
            # 计算文本尺寸
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # 旧版本PIL兼容
                text_width, text_height = draw.textsize(text, font=font)
            
            x, y = position
            img_height, img_width = image.shape[:2]
            
            # 调整位置确保文本不会超出图像边界
            if x + text_width > img_width:
                x = img_width - text_width - 5
            
            if y - text_height < 0:
                y = text_height + 5
            
            # 绘制文本背景
            bg_color = (0, 0, 0, 180)  # 半透明黑色
            bg_image = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg_image)
            bg_draw.rectangle([x-5, y-text_height-5, x+text_width+5, y+5], fill=bg_color)
            
            # 合并背景和原图
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), bg_image).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # 绘制文本
            draw.text((x, y-text_height), text, font=font, fill=color)
            
            # 转换回OpenCV格式
            image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image_bgr
            
        except Exception as e:
            self.logger.warning(f"中文文本绘制失败，回退到OpenCV: {e}")
            return self._draw_opencv_text(image, text, position, font_size/30, color, thickness)
    
    def _draw_opencv_text(self, image, text, position, font_scale, color, thickness):
        """使用OpenCV绘制文本（不支持中文但稳定）"""
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            x, y = position
            x = max(0, min(x, image.shape[1] - text_width - 5))
            y = max(text_height + 5, min(y, image.shape[0] - 5))
            
            # 绘制文本背景
            bg_x1 = x - 2
            bg_y1 = y - text_height - 2
            bg_x2 = x + text_width + 2
            bg_y2 = y + 2
            
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(image.shape[1], bg_x2)
            bg_y2 = min(image.shape[0], bg_y2)
            
            overlay = image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), self.config.TEXT_BG_COLOR, -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            
            # 绘制文本
            cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
            
            return image
            
        except Exception as e:
            self.logger.error(f"OpenCV文本绘制失败: {e}")
            return image
    
    def _draw_face_info(self, image, bbox, person_name, similarity, confidence):
        """在图像上绘制人脸框和识别信息"""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # 确定颜色：已知-绿色，未知-蓝色
        if person_name != "未知":
            box_color = self.config.BOX_COLOR_KNOWN
            text_color = self.config.TEXT_COLOR
        else:
            box_color = self.config.BOX_COLOR_UNKNOWN
            text_color = self.config.TEXT_COLOR
        
        # 绘制人脸框
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, self.config.BOX_THICKNESS)
        
        # 在外框绘制细白边增加可读性
        cv2.rectangle(image, (x1-1, y1-1), (x2+1, y2+1), (255, 255, 255), 1)
        
        # 准备显示文本 - 只显示人名，不显示相似度
        display_text = person_name
        
        # 计算文本位置（在人脸框上方）
        text_x = max(10, x1)
        text_y = max(30, y1 - 10)
        
        # 如果上方空间不足，放在框内
        if y1 < 40:
            text_y = y1 + 30
        
        # 使用指定字体绘制中文文本
        image = self._draw_chinese_text(
            image, display_text, (text_x, text_y), 
            self.config.FONT_SIZE, text_color, self.config.TEXT_THICKNESS
        )
        
        return image
    
    def _process_single_image(self, image_path):
        """处理单张图片并进行人脸识别"""
        try:
            # 加载图片
            image_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.warning(f"无法加载图片: {image_path}")
                return None, []
            
            # 检测人脸
            faces = self.app.get(image)
            
            recognition_results = []
            processed_image = image.copy()
            
            for i, face in enumerate(faces):
                confidence = face.det_score if hasattr(face, 'det_score') else face.det
                
                if confidence is None or confidence < self.config.MIN_FACE_CONFIDENCE:
                    continue
                
                # 识别人脸
                person_name, similarity = self._recognize_face(face.embedding)
                
                # 绘制识别结果
                processed_image = self._draw_face_info(processed_image, face.bbox, person_name, similarity, confidence)
                
                recognition_results.append({
                    'bbox': face.bbox.astype(int),
                    'person_name': person_name,
                    'similarity': similarity,
                    'confidence': confidence
                })
                
                # 更新统计信息
                self.recognition_stats['total_faces'] += 1
                if person_name != "未知":
                    self.recognition_stats['recognized_faces'] += 1
                else:
                    self.recognition_stats['unknown_faces'] += 1
            
            return processed_image, recognition_results
            
        except Exception as e:
            self.logger.error(f"处理图片失败 {image_path}: {e}")
            return None, []
    
    def recognize_from_folder(self, unknown_folder='./unknown'):
        """从文件夹识别未知图片"""
        self.logger.info("=== 开始人脸识别 ===")
        
        unknown_path = Path(unknown_folder)
        if not unknown_path.exists():
            self.logger.error(f"未知图片文件夹不存在: {unknown_folder}")
            return False
        
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(unknown_path.glob(ext))
            image_paths.extend(unknown_path.glob(ext.upper()))
        
        image_paths = list(set([str(path) for path in image_paths]))
        
        if len(image_paths) == 0:
            self.logger.error("未找到任何图片文件")
            return False
        
        self.recognition_stats['total_images'] = len(image_paths)
        self.logger.info(f"找到 {len(image_paths)} 张待识别图片")
        
        # 处理每张图片
        for image_path in tqdm(image_paths, desc="识别进度", unit="张"):
            image_name = Path(image_path).name
            self.logger.info(f"处理图片: {image_name}")
            
            result_image, face_results = self._process_single_image(image_path)
            
            if result_image is not None:
                # 保存结果图片
                if self.config.SAVE_IMAGES:
                    output_path = Path(self.config.OUTPUT_FOLDER) / f"result_{image_name}"
                    
                    # 确保输出目录存在
                    output_path.parent.mkdir(exist_ok=True, parents=True)
                    
                    # 保存图片
                    try:
                        success = cv2.imwrite(str(output_path), result_image)
                        if not success:
                            # 尝试使用不同的质量参数
                            success = cv2.imwrite(str(output_path), result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    except Exception as e:
                        self.logger.error(f"保存图片失败 {output_path}: {e}")
                
                # 显示图片（如果配置为显示）
                if self.config.SHOW_IMAGES:
                    # 调整显示窗口大小
                    display_height = 600
                    display_width = int(result_image.shape[1] * display_height / result_image.shape[0])
                    display_image = cv2.resize(result_image, (display_width, display_height))
                    cv2.imshow('识别结果', display_image)
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                
                # 打印识别结果
                if len(face_results) > 0:
                    for i, result in enumerate(face_results):
                        status = "✓" if result['person_name'] != "未知" else "?"
                        self.logger.info(f"  {status} 人脸{i+1}: {result['person_name']} "
                                      f"(相似度: {result['similarity']:.3f})")
                else:
                    self.logger.info("  ⚠ 未检测到人脸")
        
        # 计算处理时间
        processing_time = (datetime.now() - self.recognition_stats['start_time']).total_seconds()
        self.recognition_stats['processing_time'] = processing_time
        
        # 输出统计信息
        self.logger.info("=== 识别完成 ===")
        self.logger.info(f"处理图片数: {self.recognition_stats['total_images']}")
        self.logger.info(f"检测人脸数: {self.recognition_stats['total_faces']}")
        self.logger.info(f"识别成功: {self.recognition_stats['recognized_faces']}")
        self.logger.info(f"未知人脸: {self.recognition_stats['unknown_faces']}")
        self.logger.info(f"处理时间: {processing_time:.2f} 秒")
        self.logger.info(f"ONNX Runtime GPU加速: {'是' if self.recognition_stats['gpu_available'] else '否'}")
        self.logger.info(f"结果保存至: {self.config.OUTPUT_FOLDER}")
        
        return True

# ============================ 主函数 ============================
def main():
    """主执行函数"""
    config = RecognizeConfig()
    
    try:
        # 初始化识别器
        recognizer = FaceRecognizer(config)
        
        # 加载人脸数据库
        if not recognizer.load_database('face_database.pkl'):
            print("❌ 数据库加载失败，请先运行训练程序")
            return
        
        # 开始识别
        if recognizer.recognize_from_folder('./unknown'):
            print("\n🎉 识别完成！")
            print(f"📁 识别结果保存在: {config.OUTPUT_FOLDER}")
            print(f"⚡ ONNX Runtime GPU加速: {'是' if recognizer.recognition_stats['gpu_available'] else '否'}")
            print(f"👤 识别统计:")
            print(f"   - 总图片数: {recognizer.recognition_stats['total_images']}")
            print(f"   - 总人脸数: {recognizer.recognition_stats['total_faces']}")
            print(f"   - 识别成功: {recognizer.recognition_stats['recognized_faces']}")
            print(f"   - 未知人脸: {recognizer.recognition_stats['unknown_faces']}")
            print(f"   - 处理时间: {recognizer.recognition_stats['processing_time']:.2f}秒")
        else:
            print("❌ 识别失败，请检查unknown文件夹")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()