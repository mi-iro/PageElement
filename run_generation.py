# run_generation.py
import json
import os
from tqdm import tqdm
from bootstrap import parse_args, initialize_components
from src.loaders.base_loader import PageElement

def main():
    args = parse_args()
    print(f"🚀 Starting Generation Stage for {args.benchmark}...")
    
    # 这一步其实不需要加载 heavy 的 Reranker 或 Loader 数据，但为了复用 Agent 初始化逻辑，我们简单调用
    # 实际上可以将 init_retriever=False 从而跳过加载 Reranker 模型
    agent, _ = initialize_components(args, init_retriever=False, init_generator=True)
    
    # 读取检索阶段的结果
    retrieval_file = os.path.join(args.output_dir, "retrieval_results.json" if args.generation_input is None else args.generation_input)
    if not os.path.exists(retrieval_file):
        print(f"❌ Error: Retrieval file not found at {retrieval_file}. Run run_retrieval.py first.")
        return

    with open(retrieval_file, 'r', encoding='utf-8') as f:
        data_items = json.load(f)
    
    generation_results = []
    
    print(f"Generating answers for {len(data_items)} samples...")
    for item in tqdm(data_items, desc="Generating"):
        qid = item['qid']
        query = item['query']
        
        # 反序列化 PageElement
        retrieved_elements_data = item.get('retrieved_elements', [])
        retrieved_elements = []
        for el_dict in retrieved_elements_data:
            # 过滤掉不属于 PageElement 的字段 (防止报错)
            valid_keys = PageElement.__annotations__.keys()
            filtered_dict = {k: v for k, v in el_dict.items() if k in valid_keys}
            retrieved_elements.append(PageElement(**filtered_dict))
        
        # 调用 RAGAgent 的 generate 方法
        gen_output = agent.generate(query, retrieved_elements)
        
        # 更新结果
        item['model_answer'] = gen_output['final_answer']
        item['messages'] = gen_output['messages'] # 包含图片 Base64，文件可能较大
        
        generation_results.append(item)
    
    # 保存最终结果
    output_file = os.path.join(args.output_dir, "generation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generation_results, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Generation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()