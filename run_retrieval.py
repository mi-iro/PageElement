# run_retrieval.py
import json
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bootstrap import parse_args, initialize_components, save_run_config

def process_single_sample_retrieval(sample, agent, cache_dir):
    """
    单个样本的检索处理函数，支持缓存读取
    """
    qid = str(sample.qid)
    # 处理特殊字符，防止文件名非法
    safe_qid = "".join([c if c.isalnum() else "_" for c in qid])
    cache_path = os.path.join(cache_dir, f"{safe_qid}.json")

    # 1. Check Cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass # Cache corrupted, re-run

    # 2. Run Retrieval
    try:
        elements = agent.retrieve(sample)
        # 序列化
        elements_data = [el.to_dict() if hasattr(el, 'to_dict') else el for el in elements]
    except Exception as e:
        print(f"Error retrieving sample {qid}: {e}")
        elements_data = []

    result_item = {
        "qid": sample.qid,
        "query": sample.query,
        "gold_answer": sample.gold_answer,
        "data_source": sample.data_source,
        "gold_pages": getattr(sample, 'gold_pages', []),
        "retrieved_elements": elements_data
    }

    # 3. Save Cache
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result_item, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving cache for {qid}: {e}")

    return result_item

def main():
    args = parse_args()
    save_run_config(args, "retrieval")
    print(f"🚀 Starting Retrieval Stage for {args.benchmark} (Threads: {args.num_threads})...")
    
    # 初始化组件
    agent, loader = initialize_components(args, init_retriever=True, init_generator=False)
    
    # 准备缓存目录
    cache_dir = os.path.join(args.output_dir, "cache_retrieval_results")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"📂 Cache directory: {cache_dir}")

    retrieval_results = []
    samples = loader.samples
    print(f"Processing {len(samples)} samples...")

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # 提交所有任务
        future_to_qid = {
            executor.submit(process_single_sample_retrieval, sample, agent, cache_dir): sample.qid 
            for sample in samples
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(future_to_qid), total=len(samples), desc="Retrieving"):
            try:
                result = future.result()
                if result:
                    retrieval_results.append(result)
            except Exception as e:
                print(f"Thread exception: {e}")

    # 排序以保持顺序一致性 (多线程返回顺序是乱的)
    # 根据 qid 排序，如果 qid 不是数字，则按字符串排序
    try:
        retrieval_results.sort(key=lambda x: int(x['qid']) if str(x['qid']).isdigit() else str(x['qid']))
    except:
        pass # Fallback if mixed types

    # 保存最终汇总结果
    output_file = os.path.join(args.output_dir, "retrieval_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Retrieval complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()