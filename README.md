# PageElement: 基于视觉大模型的精细化文档元素提取与 RAG 系统

**PageElement** 是一个先进的多模态文档分析与问答框架。它利用视觉语言模型（VLM）的强大能力，结合 Agentic RAG（代理式检索增强生成）范式，实现了对文档图像的深度理解、精细化证据提取（BBox级别）以及精准的问答。

该项目特别擅长处理包含复杂排版、图表、表格的长文档，通过“**检索-重排序-精提**”的流水线，解决了传统 RAG 难以定位视觉细节的问题。

## ✨ 核心特性

* **多模态检索与重排序 (Multimodal Retrieval & Reranking)**:
* 实现了基于 `Qwen3-VL-Embedding` 模型，支持图像和文本的联合向量化。
* 实现了基于 `Qwen3-VL-Reranker` 模型，通过计算图文相关性得分进行精准重排。
* 集成 `FAISS` HNSW 索引，支持大规模文档库的高效检索。


* **Agentic RAG (代理式 RAG)**:
* 基于 ReAct (Reasoning + Acting) 范式的 `AgenticRAGAgent`，能够自动规划搜索路径，分解复杂问题。
* 支持多轮对话，动态调用检索工具获取证据。


* **精细化元素提取 (Fine-grained Element Extraction)**:
* **`ElementExtractor` Agent**: 专注于在单页文档中定位并提取具体的证据（文字、表格、图表）。
* **`ImageZoomOCRTool`**: 独创的视觉工具，支持对文档特定区域进行裁剪（Crop）、旋转和高精度 OCR（基于 MinerU），解决了小字体或密集信息的识别问题。


* **多数据集支持**:
* 内置 `FinRAG`、`MMLongBench-Doc`、`MVToolBench` 等多个高难度文档理解基准的加载器与处理流程。


## 📂 目录结构

```text
PageElement/
├── scripts/                    # 核心模型脚本
│   ├── qwen3_vl_embedding.py   # Qwen3-VL Embedding 模型实现
│   ├── qwen3_vl_reranker.py    # Qwen3-VL Reranker (打分) 模型实现
│   └── vllm.sh                 # 启动 vLLM 推理服务的脚本 (Embedding/Reranker用)
├── src/
│   ├── agents/                 # Agent 与工具实现
│   │   ├── AgenticRAGAgent.py  # 基于 ReAct 范式的上层调度 Agent
│   │   ├── ElementExtractor.py # 专注于页面内元素提取的 Agent
│   │   ├── RAGAgent.py         # 基础 RAG Agent 类
│   │   ├── utils.py            # 工具库 (ImageZoomOCRTool, MinerUClient, 坐标映射等)
│   │   └── vllm.sh             # 启动 Agent 所需的 LLM 服务脚本
│   └── loaders/                # 数据集加载与 Pipeline 定义
│       ├── base_loader.py      # 数据加载基类与数据结构定义
│       ├── FinRAGLoader.py     # FinRAG 数据集加载器 (含完整 RAG Pipeline)
│       ├── MMLongLoader.py     # MMLongBench 加载器 (含 PDF 转图片逻辑)
│       └── MVToolLoader.py     # MVToolBench 加载器
├── workspace/                  # (自动生成) 运行时产生的裁剪图片、PDF 缓存等
└── .gitignore

```

## 🛠️ 环境依赖与安装

请确保你的环境支持 PyTorch 和 CUDA。

1. **基础依赖**:
```bash
pip install torch torchvision numpy pandas scipy pillow
pip install transformers accelerate

```


2. **RAG 与 向量库**:
```bash
pip install faiss-cpu  # 或 faiss-gpu

```


3. **大模型推理**:
```bash
pip install vllm openai
pip install qwen-vl-utils

```


4. **OCR 与 视觉工具 (MinerU)**:
你需要安装 MinerU 相关的客户端库或确保 `mineru_vl_utils` 可用。
* 项目依赖 `MinerU` 进行高精度 OCR 和布局分析，请参考 [MinerU 官方仓库](https://github.com/opendatalab/MinerU) 。



## 🚀 快速开始

### 1. 模型准备

本项目依赖 `Qwen3-VL` (或 Qwen2.5-VL) 系列模型作为基座。请下载以下模型权重：

* **Embedding 模型**: 用于文档向量化。
* **Reranker 模型**: 用于检索结果重排序。
* **Instruct 模型**: 用于 Agent 对话与逻辑推理。

### 2. 启动推理服务 (vLLM)

Agent 需要通过 OpenAI 兼容接口调用 LLM。使用提供的脚本启动 vLLM 服务：

```bash
# 启动用于 Agent 的推理服务 (默认端口 8000)
bash src/agents/vllm.sh

```

*注意：请修改 `src/agents/vllm.sh` 中的 `--model` 路径为你本地的模型路径。*

### 3. 配置 MinerU OCR 服务

`ImageZoomOCRTool` 依赖 MinerU 服务进行 OCR。你需要确保 MinerU API 服务已启动，并在代码中配置正确的 `mineru_server_url`。

### 4. 运行 Pipeline 示例

#### A. 运行 FinRAG 流程 (检索 -> 重排 -> 提取)

`FinRAGLoader` 展示了完整的 RAG 流程。

1. 修改 `src/loaders/FinRAGLoader.py` 中的 `main` 部分：
* 配置 `embedding_model_path` 和 `reranker_model_path`。
* 配置 `root_dir` 指向你的 FinRAG 数据集。
* 配置 `api_key` 和 `base_url` 指向你的 vLLM 服务。


2. 运行脚本：
```bash
python src/loaders/FinRAGLoader.py

```


*首次运行会建立 HNSW 索引，耗时较长。*

#### B. 运行 Agentic RAG (多轮推理)

使用 `AgenticRAGAgent` 处理复杂问题。

1. 修改 `src/agents/AgenticRAGAgent.py` 中的测试代码：
* 指定数据集路径和 LLM API 配置。


2. 运行脚本：
```bash
python src/agents/AgenticRAGAgent.py

```



## 🧩 模块详解

### 1. 数据加载与 Pipeline (`src/loaders`)

* **FinRAGLoader**:
* **Step 1**: 使用 `Qwen3VLEmbedder` 将 Query 和文档图像转为向量。
* **Step 2**: 使用 FAISS 进行粗排检索 (Top-K)。
* **Step 3**: 使用 `Qwen3VLReranker` 对候选图进行精细重排。
* **Step 4**: 调用 `ElementExtractor` 从 Top-N 图片中提取具体证据。


* **MMLongLoader**: 处理长文档 PDF，自动将 PDF 转为图片序列，并逐页进行证据提取。

### 2. 智能体 (`src/agents`)

* **ElementExtractor**:
* 核心 Prompt 定义在 `SYSTEM_PROMPT` 中，指导模型输出 JSON 格式的 BBox。
* 循环调用 `image_zoom_and_ocr_tool`，通过“看大图 -> 切小图 -> OCR”的流程获取精准信息。


* **AgenticRAGAgent**:
* 使用 ReAct 模式。当用户问题复杂时，模型会思考 (`Thought`) 并生成 `<tool_call>` 来调用 `search_evidence_tool`，根据检索结果更新回答。



### 3. 视觉增强工具 (`src/agents/utils.py`)

* **ImageZoomOCRTool**:
* **Smart Resize**: 智能缩放图片以适应模型输入的最佳分辨率。
* **Coordinate Mapping**: 处理裁剪后的坐标变换，将 OCR 结果的局部坐标映射回原图的全局坐标 (0-1000 归一化坐标)，确保 BBox 的准确性。



## ⚠️ 注意事项

1. **路径配置**: 代码中含有大量硬编码的绝对路径 (如 `/mnt/shared-storage-user/...`)。在运行前，**务必**全局搜索并替换为你本地的数据集和模型路径。
2. **API Key**: 代码示例中使用了占位符 API Key (`sk-123456`)，请替换为你实际启动的 vLLM 或 OpenAI 服务 Key。
3. **PDF 处理**: `MMLongLoader` 依赖 `pdf2image`，请确保系统安装了 `poppler-utils`。