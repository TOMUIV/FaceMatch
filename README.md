# 🎯 FaceMatch - 人脸识别体验工具

✨ **感受人脸识别的魅力！** ✨

FaceMatch 是一个基于 [InsightFace](https://github.com/deepinsight/insightface) 的轻量级人脸识别体验工具。只需几步，你就能体验从人脸注册到实时识别的完整流程！

## 🎮 三步玩转人脸识别

```
1️⃣ 注册人脸 → 2️⃣ 图片识别 → 3️⃣ 实时视频识别
```

## ✨ 功能特性

- 📝 **人脸注册** - 上传照片，让电脑"记住"你的脸
- 🖼️ **图片识别** - 批量识别图片，自动标注姓名
- 📹 **实时视频** - 打开摄像头，体验实时人脸识别
- 🎯 **高精度** - 使用 InsightFace `buffalo_l` 模型
- 🇨🇳 **中文支持** - 完美支持中文姓名显示
- ⚡ **GPU加速** - 有显卡自动加速，无显卡也能跑

## 📁 项目结构

```
.
├── 🏋️ train.py              # 📝 人脸注册
├── 🔍 pic_recognize.py      # 🖼️ 图片识别
├── 📹 webcam_recognize.py   # 📹 实时视频识别
├── 📄 templates/
│   └── 🎨 index.html        # 🌐 网页界面
├── 📂 data/
│   ├── traindata/          # 👤 人脸照片（按人分文件夹）
│   └── unknown/            # ❓ 待识别图片
├── 📂 output/               # 🎉 识别结果
├── 📝 requirements.txt      # 📦 Python依赖
├── 🅰️ msyh.ttc              # 🔤 中文字体
└── 📖 README.md             # 📚 本文件
```

## 🛠️ 环境要求

- 🐍 Python 3.7+
- 🎮 NVIDIA GPU（可选，有的话更快）

## 📦 快速开始

```bash
# 1️⃣ 克隆仓库
git clone https://github.com/TOMUIV/FaceMatch.git
cd FaceMatch

# 2️⃣ 创建环境
conda create -n facematch python=3.9
conda activate facematch

# 3️⃣ 安装依赖
pip install -r requirements.txt

# 4️⃣ 准备字体
# 确保 msyh.ttc 在项目根目录（用于显示中文）
```

## 🚀 使用指南

### 📝 第一步：人脸注册

把人脸照片按这个结构放：

```
data/traindata/
├── 😎 张三/
│   ├── photo1.jpg
│   └── photo2.jpg
├── 🥳 李四/
│   └── photo1.jpg
```

然后运行：
```bash
python train.py
```

💡 **提示**：文件夹名就是识别出来的名字！

---

### 🖼️ 第二步：图片识别

把要识别的图片放 `data/unknown`，然后：

```bash
python pic_recognize.py
```

🎉 结果在 `output/` 文件夹，打开就能看到标注好的图片！

---

### 📹 第三步：实时视频识别（最酷的部分！）

```bash
python webcam_recognize.py
```

浏览器打开 👉 http://localhost:5000

点击"启动摄像头"，体验实时人脸识别！😎

> ⚠️ **小提示**：WebRTC 在某些浏览器可能不稳定，建议用 Chrome 或 Edge

---

## ⚙️ 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `SIMILARITY_THRESHOLD` | `0.6` | 相似度阈值（越高越严格） |
| `MIN_FACE_CONFIDENCE` | `0.3` | 人脸检测置信度 |
| `PORT` | `5000` | 网页服务端口号 |

## 🛠️ 技术栈

- 🤖 [InsightFace](https://github.com/deepinsight/insightface) - 人脸识别框架
- ⚡ ONNX Runtime - 模型推理
- 🌶️ Flask + SocketIO - Web服务
- 📷 OpenCV - 图像处理
- 🎨 HTML5 + WebRTC - 实时视频

## ⚠️ 注意事项

1. 📥 **首次运行**会下载模型（约 100MB），请耐心等待
2. 🎮 **GPU 加速**需要 CUDA，没有会自动用 CPU
3. 🅰️ **字体文件** `msyh.ttc` 必须存在，否则中文乱码
4. 📸 **照片质量**影响识别效果，建议用清晰正面照

## 🎉 玩得开心！

有问题欢迎提 Issue ~ 😊

---

⭐ **如果觉得好用，给个 Star 吧！**
