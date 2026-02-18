#  InsightFace 人脸识别工具

基于 [InsightFace](https://github.com/deepinsight/insightface) 的轻量级人脸识别工具，支持人脸注册和批量图片识别。

##  功能特性

-  **人脸注册** - 从照片提取人脸特征，构建人脸数据库
-  **图片识别** - 批量识别图片中的人脸，自动标注姓名
-  **高精度识别** - 使用 InsightFace `buffalo_l` 模型
-  **中文支持** - 完美支持中文姓名显示
-  **GPU加速** - 自动检测并使用 NVIDIA GPU 加速

##  项目结构

```
.
  train.py              # 人脸注册：构建人脸数据库
  pic_recognize.py      # 图片识别：批量识别未知图片
  data/
    traindata/          # 训练数据（每人一个文件夹）
    unknown/            # 待识别图片
  output/               # 识别结果输出
  requirements.txt      # Python依赖
  msyh.ttc              # 微软雅黑字体
  README.md             # 本文件
```

##  环境要求

-  Python 3.7+
-  NVIDIA GPU（可选，用于加速）

##  安装

```bash
# 1. 克隆仓库
git clone https://github.com/TOMUIV/face_recognition_based_on_insightface.git
cd face_recognition_based_on_insightface

# 2. 创建虚拟环境
conda create -n face python=3.9
conda activate face

# 3. 安装依赖
pip install -r requirements.txt
```

##  使用方法

### 第一步：准备训练数据

将人脸照片按以下结构放入 `data/traindata`：

```
data/traindata/
 张三/
    photo1.jpg
    photo2.jpg
 李四/
    photo1.jpg
```

**注意**：文件夹名就是识别出来的名字！

### 第二步：人脸注册

```bash
python train.py
```

生成 `face_database.pkl` 人脸数据库。

### 第三步：图片识别

将待识别图片放入 `data/unknown`，然后运行：

```bash
python pic_recognize.py
```

结果保存在 `output/` 文件夹。

##  注意事项

1.  **首次运行**会自动下载 InsightFace 模型（约 100MB）
2.  **GPU 加速**需要安装 CUDA，没有会自动使用 CPU
3.  确保 `msyh.ttc` 字体文件存在，否则中文显示会乱码

##  许可证

本项目仅供学习研究使用。

---

 **玩得开心！有问题欢迎提 Issue~**
