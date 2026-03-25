import cv2
import numpy as np
import random
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime

from safe_storage import DEFAULT_DATABASE_PATH, save_face_database

# ============================ 全局配置参数 ============================
class Config:
    # 模型配置
    MODEL_NAME = 'buffalo_l'
    DET_SIZE = (640, 640)
    
    # 数据处理配置
    MAX_IMAGES_PER_PERSON = 50
    BATCH_SIZE = 10
    SHUFFLE_IMAGES = True
    MIN_FACE_CONFIDENCE = 0.3
    
    # CUDA配置
    FORCE_CPU = False
    CUDA_DEVICE_ID = 0
    
    # 日志配置
    SUPPRESS_MODEL_LOGS = True
    SHOW_IMAGE_DETAILS = False

# ============================ 修复版人脸训练器 ============================
class FixedFaceTrainer:
    def __init__(self, config):
        self.config = config
        self.face_database = {}
        self.processing_stats = {
            'total_persons': 0,
            'total_images': 0,
            'total_faces': 0,
            'successful_persons': 0,
            'failed_persons': 0,
            'processing_time': 0,
            'cuda_available': False,
            'start_time': datetime.now()
        }
        self._setup_logging()
        self._initialize_model()
    
    def _setup_logging(self):
        """设置日志"""
        if self.config.SUPPRESS_MODEL_LOGS:
            logging.getLogger('insightface').setLevel(logging.WARNING)
            logging.getLogger('onnxruntime').setLevel(logging.WARNING)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self):
        """初始化人脸分析模型"""
        self.logger.info("=== 模型初始化 ===")
        
        try:
            # 临时重定向stdout来抑制模型加载日志
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            from insightface.app import FaceAnalysis
            
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            ctx_id = self.config.CUDA_DEVICE_ID
            
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
            raise
    
    def _get_image_paths(self, folder_path):
        """获取文件夹中的所有图片路径"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(folder_path.glob(ext))
            image_paths.extend(folder_path.glob(ext.upper()))
        
        image_paths = list(set([str(path) for path in image_paths]))
        
        if self.config.SHUFFLE_IMAGES:
            random.shuffle(image_paths)
        
        if len(image_paths) > self.config.MAX_IMAGES_PER_PERSON:
            image_paths = image_paths[:self.config.MAX_IMAGES_PER_PERSON]
        
        return image_paths
    
    def _load_image(self, image_path):
        """加载图片"""
        try:
            image_array = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
                
            return image
            
        except Exception as e:
            return None
    
    def _process_single_image(self, image_path):
        """处理单张图片"""
        image = self._load_image(image_path)
        if image is None:
            return []
        
        try:
            faces = self.app.get(image)
            
            results = []
            for face in faces:
                confidence = face.det_score if hasattr(face, 'det_score') else face.det
                
                if confidence is None or confidence < self.config.MIN_FACE_CONFIDENCE:
                    continue
                
                bbox = face.bbox.astype(int)
                embedding = face.embedding
                
                results.append({
                    'bbox': bbox,
                    'embedding': embedding,
                    'confidence': confidence,
                    'image_path': image_path
                })
            
            return results
            
        except Exception as e:
            return []
    
    def _process_person(self, person_name, image_paths):
        """处理单个人 - 修复进度条问题"""
        self.logger.info(f"开始处理 {person_name} 的 {len(image_paths)} 张图片")
        
        all_faces = []
        
        # 使用更简单的进度条实现
        pbar = tqdm(total=len(image_paths), desc=f"处理 {person_name}", unit="张")
        
        for i in range(0, len(image_paths), self.config.BATCH_SIZE):
            batch_paths = image_paths[i:i + self.config.BATCH_SIZE]
            
            for image_path in batch_paths:
                faces = self._process_single_image(image_path)
                all_faces.extend(faces)
                pbar.update(1)
                
                # 手动更新进度条描述，避免使用set_postfix
                pbar.set_description(f"处理 {person_name} (有效人脸: {len(all_faces)})")
        
        pbar.close()
        return all_faces
    
    def calculate_average_embedding(self, embeddings):
        """计算平均特征向量"""
        if not embeddings:
            return None
        
        embedding_array = np.array(embeddings)
        average_embedding = np.mean(embedding_array, axis=0)
        average_embedding = average_embedding / np.linalg.norm(average_embedding)
        
        return average_embedding
    
    def train_from_folder(self, data_folder='./traindata'):
        """从文件夹训练人脸数据"""
        self.logger.info("=== 开始人脸数据训练 ===")
        
        data_path = Path(data_folder)
        if not data_path.exists():
            self.logger.error(f"数据文件夹不存在: {data_folder}")
            return False
        
        person_folders = [f for f in data_path.iterdir() if f.is_dir()]
        self.processing_stats['total_persons'] = len(person_folders)
        
        if len(person_folders) == 0:
            self.logger.error("未找到任何人脸数据文件夹")
            return False
        
        self.logger.info(f"找到 {len(person_folders)} 个人的数据")
        
        # 处理每个人
        for person_folder in person_folders:
            person_name = person_folder.name
            self.logger.info(f"处理: {person_name}")
            
            image_paths = self._get_image_paths(person_folder)
            
            if len(image_paths) == 0:
                self.logger.warning(f"{person_name} 文件夹中没有找到图片")
                self.processing_stats['failed_persons'] += 1
                continue
            
            self.processing_stats['total_images'] += len(image_paths)
            
            # 处理图片
            face_results = self._process_person(person_name, image_paths)
            
            if len(face_results) > 0:
                embeddings = [result['embedding'] for result in face_results]
                avg_embedding = self.calculate_average_embedding(embeddings)
                
                if avg_embedding is not None:
                    self.face_database[person_name] = {
                        'embedding': avg_embedding,
                        'sample_count': len(embeddings),
                        'image_count': len(image_paths)
                    }
                    
                    self.processing_stats['total_faces'] += len(embeddings)
                    self.processing_stats['successful_persons'] += 1
                    self.logger.info(f"成功注册 {person_name}: {len(embeddings)} 张人脸")
                else:
                    self.logger.warning(f"无法为 {person_name} 计算特征向量")
                    self.processing_stats['failed_persons'] += 1
            else:
                self.logger.warning(f"{person_name} 没有检测到有效人脸")
                self.processing_stats['failed_persons'] += 1
        
        processing_time = (datetime.now() - self.processing_stats['start_time']).total_seconds()
        self.processing_stats['processing_time'] = processing_time
        
        self.logger.info("=== 训练完成 ===")
        self.logger.info(f"成功注册: {self.processing_stats['successful_persons']} 人")
        self.logger.info(f"失败: {self.processing_stats['failed_persons']} 人")
        self.logger.info(f"总人脸数: {self.processing_stats['total_faces']}")
        self.logger.info(f"处理时间: {processing_time:.2f} 秒")
        
        return self.processing_stats['successful_persons'] > 0
    
    def save_database(self, save_path=DEFAULT_DATABASE_PATH):
        """保存人脸数据库"""
        try:
            saved_path = save_face_database(
                self.face_database,
                processing_stats=self.processing_stats,
                config=self.config,
                save_path=save_path
            )
            self.logger.info(f"人脸数据库已保存: {saved_path}")
            
        except Exception as e:
            self.logger.error(f"保存数据库失败: {e}")

# ============================ 主函数 ============================
def main():
    """主执行函数"""
    config = Config()
    
    try:
        trainer = FixedFaceTrainer(config)
        
        if trainer.train_from_folder('./traindata'):
            trainer.save_database(DEFAULT_DATABASE_PATH)
            
            print("\n🎉 训练完成！")
            print(f"📁 所有人脸数据保存在: {DEFAULT_DATABASE_PATH}")
        else:
            print("❌ 训练失败，请检查数据路径和格式")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
