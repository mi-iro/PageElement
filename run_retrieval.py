# run_retrieval.py
import json
import os
from tqdm import tqdm
from bootstrap import parse_args, initialize_components

def main():
    args = parse_args()
    print(f"🚀 Starting Retrieval Stage for {args.benchmark}...")
    
    # 初始化组件 (需要 Retriever)
    agent, loader = initialize_components(args, init_retriever=True, init_generator=False)
    
    retrieval_results = []
    
    print(f"Processing {len(loader.samples)} samples...")
    for sample in tqdm(loader.samples, desc="Retrieving"):
        # 调用 RAGAgent 的 retrieve 方法
        elements = agent.retrieve(sample)
        
        # 将 PageElement 对象转换为可序列化的字典
        elements_data = [el.to_dict() if hasattr(el, 'to_dict') else el for el in elements]
        
        result_item = {
            "qid": sample.qid,
            "query": sample.query,
            "gold_answer": sample.gold_answer,
            "data_source": sample.data_source,
            "gold_pages": getattr(sample, 'gold_pages', []),
            "retrieved_elements": elements_data
        }
        retrieval_results.append(result_item)
    
    # 保存中间结果
    output_file = os.path.join(args.output_dir, "retrieval_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Retrieval complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()