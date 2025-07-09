# GLM-4V 图片描述生成器

![GLM-4V Logo](https://img.shields.io/badge/GLM--4V-Image%20Descriptor-blue?style=for-the-badge&logo=openai)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-green?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-purple?style=flat-square)](https://github.com/comfyanonymous/ComfyUI)

> 🚀 **专业的AI图片描述生成工具**  
> 基于智谱AI GLM-4V多模态模型，为图片批量生成精确、详细的中英文描述

## ✨ 核心特性

- 🎯 **智能批量处理** - 一键处理整个文件夹的图片
- 📝 **同名文件输出** - 自动生成与图片同名的txt描述文件
- ⚡ **高性能优化** - 支持4bit量化，GPU加速，内存管理优化
- 🔧 **灵活配置** - 可自定义提示词、模型选择、输出格式
- 🌐 **双语支持** - 支持中英文界面和文档
- 📊 **多格式输出** - 支持TXT、JSON、CSV等多种输出格式
- 🔌 **ComfyUI集成** - 完全兼容ComfyUI工作流

## 🖼️ 效果展示

### 输出示例
**输入图片**: `sunset_beach.jpg`  
**输出文件**: `sunset_beach.txt`  
**内容**: "A breathtaking sunset scene over a serene beach with golden sand, where gentle waves lap against the shore while vibrant orange and pink hues paint the sky, creating a peaceful and romantic atmosphere."

## 🚀 快速开始

### 环境要求

- **Python**: 3.8 或更高版本
- **GPU**: 推荐 8GB+ 显存的NVIDIA显卡
- **系统**: Windows / Linux / macOS
- **内存**: 16GB+ 系统内存

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-username/ComfyUI_GLM4V_voltspark.git
cd ComfyUI_GLM4V_voltspark
```

2. **安装依赖**
```bash
pip install -r requirements.txt

# 升级transformers库到最新版本（推荐）
python -m pip install git+https://github.com/huggingface/transformers.git

# 或者安装指定版本
pip install transformers==4.54.0
```

3. **模型下载和安装**

### 方式一：国内网盘下载（推荐）

**glmv4_4bit 模型：**
- 📁 下载地址：[百度网盘](https://pan.baidu.com/s/1x1vWW09YadUdz1EYPWc-Fg?pwd=qbdq)
- 🔑 提取码：`qbdq`
- 📦 文件名：`glmv4_4bit.7z`
- 📂 解压路径：`ComfyUI/models/glmv4_4bit/`

**GLM-4.1V-9B-Thinking 模型：**
- 📁 下载地址：[百度网盘](https://pan.baidu.com/s/1xXtfKXEJLKg2iJ86OZR6nw?pwd=9n27)
- 🔑 提取码：`9n27`
- 📦 文件名：`GLM-4.1V-9B-Thinking.rar`
- 📂 解压路径：`ComfyUI/models/GLM-4.1V-9B-Thinking/`

### 方式二：自动下载
模型会在首次运行时自动从Hugging Face下载，确保网络连接正常。

4. **插件安装**
```bash
# 将插件解压到ComfyUI的custom_nodes目录下
ComfyUI/custom_nodes/ComfyUI_GLM4V_voltspark/
```

### 📂 完整安装目录结构

安装完成后，您的目录结构应该如下：

```
ComfyUI/
├── models/
│   ├── glmv4_4bit/                    # GLM-4V 4bit量化模型目录
│   │   ├── config.json
│   │   ├── modeling_chatglm.py
│   │   ├── pytorch_model-*.bin
│   │   └── ...
│   └── GLM-4.1V-9B-Thinking/          # GLM-4.1V完整模型目录
│       ├── config.json
│       ├── modeling_chatglm.py
│       ├── pytorch_model-*.bin
│       └── ...
└── custom_nodes/
    └── ComfyUI_GLM4V_voltspark/       # 本插件目录
        ├── glm4v.py
        ├── __init__.py
        ├── requirements.txt
        ├── README.md
        └── ...
```

### 使用方法

#### ComfyUI节点使用

1. **节点类型**
   - `GLM-4V Generate` - 单张图片处理
   - `GLM-4V Batch Generate` - 批量图片处理

2. **使用步骤**
   - 在ComfyUI中搜索"GLM-4V"
   - 将节点添加到工作流
   - 配置输入参数
   - 执行工作流

## 📋 详细配置

### 预置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| **提示词** | `describe this image,Describe in long sentence form, without using Markdown format.` | 预设的优化提示词 |
| **模型** | `glmv4_4bit` | 4bit量化GLM-4V模型 |
| **卸载策略** | `Never` | 模型保持加载状态 |
| **输出格式** | `TXT` | 纯文本格式输出 |
| **最大处理数** | `100` | 单次处理的最大图片数量 |

### 支持的图片格式

- 📷 **常用格式**: JPG, JPEG, PNG, BMP
- 🎨 **专业格式**: TIFF, WEBP
- 📐 **分辨率**: 支持各种分辨率，自动优化处理

### 模型选择

| 模型名称 | 大小 | 特点 | 推荐用途 |
|----------|------|------|----------|
| `glmv4_4bit` | ~6.7GB | 4bit量化，速度快 | 日常批量处理 |
| `GLM-4.1V-9B-Thinking` | ~8.0GB | 完整精度，质量高 | 高质量描述 |

## 🛠️ 高级功能

### 自定义提示词

您可以根据需求自定义提示词：

```python
# 详细描述模式
"Please provide a detailed description of this image, including objects, colors, composition, mood, and artistic style."

# 简洁描述模式  
"Describe this image briefly and accurately."

# 专业摄影模式
"Analyze this image from a photographer's perspective, describing composition, lighting, and technical aspects."
```

### 批量处理选项

- ✅ **自动保存** - 为每张图片生成同名txt文件
- 🔄 **覆盖模式** - 选择是否覆盖已存在的文件
- 📊 **进度监控** - 实时显示处理进度和统计信息
- 🛑 **中断恢复** - 支持暂停和恢复处理

### 输出格式选择

1. **TXT格式** - 纯文本描述，与图片同名
2. **JSON格式** - 结构化数据，包含元信息
3. **CSV格式** - 表格数据，便于数据分析

## 📁 项目结构

```
ComfyUI_GLM4V_voltspark/
├── 📄 glm4v.py                     # ComfyUI节点核心实现
├── 🔧 requirements.txt             # Python依赖包列表
├── 🔌 __init__.py                  # ComfyUI插件注册文件
├── 📖 使用说明.md                  # 详细使用说明文档
├── 📖 README_CN.md                 # 中文说明文档
├── 📖 README.md                    # 英文说明文档
└── 📁 Example/                     # 示例文件和工作流
    ├── 单图反推-批量打标.json        # ComfyUI工作流示例
    └── 单图反推-批量打标.png         # 工作流截图
```

## 🐛 故障排除

### 常见问题及解决方案

#### Q: 程序启动失败
**A**: 检查Python版本和依赖安装
```bash
python --version  # 确保Python 3.8+
pip install -r requirements.txt --upgrade
```

#### Q: 模型下载失败
**A**: 检查网络连接，可尝试使用镜像源
```bash
# 设置Hugging Face镜像
export HF_ENDPOINT=https://hf-mirror.com
```

#### Q: GPU内存不足
**A**: 使用4bit量化模型或调整批处理大小
- 选择 `glmv4_4bit` 模型
- 减少最大处理图片数量
- 关闭其他GPU应用程序

#### Q: 处理速度慢
**A**: 优化设置提升性能
- 确保使用GPU加速
- 设置卸载策略为"Never"
- 检查CUDA驱动版本

### 日志分析

程序运行时会显示详细日志信息：
- ✅ **成功**: 绿色状态，操作正常
- ⚠️ **警告**: 黄色状态，需要注意
- ❌ **错误**: 红色状态，需要处理

## 🤝 贡献指南

我们欢迎社区贡献！参与方式：

1. 🍴 Fork 项目仓库
2. 🔧 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 💾 提交更改 (`git commit -m 'Add amazing feature'`)
4. 📤 推送分支 (`git push origin feature/amazing-feature`)
5. 🔄 创建 Pull Request

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install jupyter matplotlib tqdm  # 可选开发工具

# 运行测试
python -m pytest tests/  # 如果有测试文件

# 代码格式化
black . --line-length 88
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [智谱AI](https://www.zhipuai.cn/) - 提供GLM-4V多模态模型
- [Hugging Face](https://huggingface.co/) - 模型托管和推理框架
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的AI工作流平台
- 所有为开源社区做出贡献的开发者们

## 📞 联系我们

- 🐛 **Bug报告**: [Issues页面](https://github.com/your-username/ComfyUI_GLM4V_voltspark/issues)
- 💡 **功能建议**: [Discussions页面](https://github.com/your-username/ComfyUI_GLM4V_voltspark/discussions)
- 📧 **邮件联系**: your-email@example.com

## 📈 更新日志

### v0.3.42 (最新版本)
- ✅ 完善的GUI界面设计
- ✅ 优化的批量处理性能
- ✅ 稳定的模型加载机制
- ✅ 完整的中英文文档

### 即将发布
- 🔄 更多模型支持
- 🎨 界面主题定制
- 📊 高级数据分析功能
- 🌐 多语言支持扩展

---

<div align="center">
  <p><strong>⭐ 如果这个项目对您有帮助，请给我们一个星标支持！ ⭐</strong></p>
  <p>Made with ❤️ by the Community</p>
</div> 