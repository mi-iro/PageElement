import argparse
import os
import sys
import torch
import asyncio

# 确保项目根目录在 sys.path 中，以便导入 src 模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.agents.AgenticRAGAgent import AgenticRAGAgent
from src.agents.ElementExtractor import ElementExtractor
from src.agents.utils import ImageZoomOCRTool
from src.loaders.MMLongLoader import MMLongLoader
from src.loaders.FinRAGLoader import FinRAGLoader
from src.utils.llm_helper import create_llm_caller

try:
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    from scripts.qwen3_vl_reranker import Qwen3VLReranker
except ImportError:
    print("Warning: Qwen3 VL scripts not found.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run AgenticRAGAgent benchmark evaluation.")

    # 基础配置
    parser.add_argument("--benchmark", type=str, default="mmlong", choices=["mmlong", "finrag"], help="Target benchmark to run.")
    parser.add_argument("--data_root", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/MMLongBench-Doc", help="Path to the dataset root directory.")
    parser.add_argument("--output_dir", type=str, default="./results_mmlong", help="Directory to save results and cache.")
    
    # LLM 配置 (Main Agent 使用)
    parser.add_argument("--model_name", type=str, default="qwen3-max", help="LLM model name (e.g., qwen3-max).")
    parser.add_argument("--base_url", type=str, default="http://localhost:3888/v1", help="LLM API Base URL.")
    parser.add_argument("--api_key", type=str, default="sk-6TGzZJkJ5HfZKwnrS1A1pMb1lH5D7EDfSVC6USq24aN2JaaR", help="LLM API Key.")
    
    # Agent 配置
    parser.add_argument("--max_rounds", type=int, default=5, help="Maximum thinking rounds for AgenticRAGAgent.")
    parser.add_argument("--limit", type=int, default=5, help="Limit the number of samples for testing (e.g., 10).")

    # 模型路径配置 (根据 Benchmark 需求)
    parser.add_argument("--embedding_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B", help="Path to Qwen3-VL-Embedding model (Required for FinRAG).")
    parser.add_argument("--reranker_model", type=str, default="/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B", help="Path to Qwen3-VL-Reranker model.")
    
    # MinerU / OCR 配置
    parser.add_argument("--mineru_server_url", type=str, default="http://10.102.250.36:8000/", help="MinerU API Server URL.")
    parser.add_argument("--mineru_model_path", type=str, default="/root/checkpoints/MinerU2.5-2509-1.2B/", help="MinerU Model Path.")
    
    # ElementExtractor 配置
    parser.add_argument("--extractor_model_name", type=str, default="MinerU-Agent-CK300", help="LLM model name (e.g., qwen3-max).")
    parser.add_argument("--extractor_base_url", type=str, default="http://localhost:8001/v1", help="LLM API Base URL.")
    parser.add_argument("--extractor_api_key", type=str, default="sk-123456", help="LLM API Key.")

    # FinRAG 特有配置
    parser.add_argument("--finrag_lang", type=str, default="ch", choices=["ch", "en", "bbox"], help="Language/subset for FinRAG.")
    parser.add_argument("--force_rebuild_index", action="store_true", help="Force rebuild vector index for FinRAG.")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 目录准备
    os.makedirs(args.output_dir, exist_ok=True)
    workspace_dir = os.path.join(args.output_dir, "workspace")
    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"🚀 Starting Benchmark: {args.benchmark.upper()}")
    print(f"📂 Data Root: {args.data_root}")
    print(f"💾 Output Dir: {args.output_dir}")

    # 2. 初始化底层工具与提取器 (ElementExtractor)
    print("🛠️ Initializing Tools and Extractor...")
    tool = ImageZoomOCRTool(
        work_dir=os.path.join(workspace_dir, "crops"),
        mineru_server_url=args.mineru_server_url,
        mineru_model_path=args.mineru_model_path
    )
    
    # 注意：Extractor 通常使用 Vision 模型 (如 qwen-vl-max 或 qwen3-vl-instruct)
    # 这里复用 args.base_url，但你可以根据需要分离 Extractor 的 LLM 配置
    extractor = ElementExtractor(
        base_url=args.extractor_base_url,
        api_key=args.extractor_api_key,
        model_name=args.extractor_model_name,
        tool=tool
    )

    loader = None

    # 3. 初始化 DataLoader
    if args.benchmark == "mmlong":
        print("📥 Loading MMLongLoader...")
        loader = MMLongLoader(
            data_root=args.data_root, 
            extractor=extractor,
            reranker_model_path=args.reranker_model
        )
        loader.load_data()

    elif args.benchmark == "finrag":
        print("📥 Loading FinRAGLoader...")
        if not args.embedding_model or not args.reranker_model:
            raise ValueError("FinRAG benchmark requires --embedding_model and --reranker_model.")
        
        # 加载本地模型
        print("   Loading Embedding Model (this may take time)...")
        embedder = Qwen3VLEmbedder(model_name_or_path=args.embedding_model, torch_dtype=torch.float16)
        print("   Loading Reranker Model...")
        reranker = Qwen3VLReranker(model_name_or_path=args.reranker_model, torch_dtype=torch.float16)

        loader = FinRAGLoader(
            data_root=args.data_root,
            lang=args.finrag_lang,
            embedding_model=embedder,
            rerank_model=reranker,
            extractor=extractor
        )
        loader.load_data()
        
        # 建立或加载索引
        loader.build_page_vector_pool(batch_size=4, force_rebuild=args.force_rebuild_index)

    # 设置用于评估的 LLM Helper
    loader.llm_caller = create_llm_caller()

    # 4. 截取测试样本 (如果设置了 limit)
    if args.limit and args.limit > 0:
        print(f"⚠️ Limiting samples to {args.limit} for testing.")
        loader.samples = loader.samples[:args.limit]

    print(f"📊 Total Samples to Process: {len(loader.samples)}")

    # 5. 初始化 Agentic RAG Agent
    agent = AgenticRAGAgent(
        loader=loader,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        max_rounds=args.max_rounds,
        cache_dir=cache_dir
    )

    # 6. 执行处理循环
    print("\n⚡ Starting Processing Loop...")
    for i, sample in enumerate(loader.samples):
        print(f"[{i+1}/{len(loader.samples)}] Processing Sample QID: {sample.qid}")
        agent.process_sample(sample)

    # 7. 保存结果
    excel_path = os.path.join(args.output_dir, f"{args.benchmark}_results.xlsx")
    json_path = os.path.join(args.output_dir, f"{args.benchmark}_results.json")
    agent.save_results(excel_path=excel_path, json_path=json_path)

    # 8. 执行评估
    print("\n📈 Starting Evaluation...")
    try:
        metrics = loader.evaluate()
        # 保存评估指标
        metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, indent=2)
        print(f"✅ Evaluation complete. Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()