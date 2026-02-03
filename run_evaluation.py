# run_evaluation.py

import json
import os
from bootstrap import parse_args, initialize_components, save_run_config
from src.loaders.base_loader import PageElement
from src.utils.llm_helper import create_llm_caller

def main():
    args = parse_args()
    save_run_config(args, "evaluation")
    print(f"🚀 Starting Evaluation Stage for {args.benchmark} (Task: {args.evaluation_task})...")
    
    # 初始化 Loader
    _, loader = initialize_components(args, init_retriever=False, init_generator=False)
    loader.llm_caller = create_llm_caller()
    
    # 1. 确定输入文件
    input_file = args.evaluation_input
    if input_file is None:
        if args.evaluation_task == "retrieval":
            p1 = os.path.join(args.output_dir, "retrieval_results.json")
            p2 = os.path.join(args.output_dir, "generation_results.json")
            input_file = p1 if os.path.exists(p1) else p2
        else:
            input_file = os.path.join(args.output_dir, "generation_results.json")
    else:
        input_file = os.path.join(args.output_dir, input_file)
    
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return

    print(f"📂 Loading results from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
        
    # 2. 将结果映射回 Loader 的 samples
    results_map = {str(item['qid']): item for item in results_data}
    
    matched_count = 0
    # 注意：这里需要确保 qid 类型一致（str vs int）
    for sample in loader.samples:
        s_qid = str(sample.qid)
        if s_qid in results_map:
            res = results_map[s_qid]
            if sample.extra_info is None:
                sample.extra_info = {}
            
            if 'retrieved_elements' in res:
                sample.extra_info['retrieved_elements'] = res['retrieved_elements']
            
            if 'model_answer' in res:
                sample.extra_info['final_answer'] = res['model_answer']
            elif 'final_answer' in res: 
                sample.extra_info['final_answer'] = res['final_answer']

            matched_count += 1
            
    print(f"✅ Mapped results for {matched_count}/{len(loader.samples)} samples.")
    
    final_metrics = {}

    # 3. 执行评估
    # 注意：评估通常涉及全集统计计算（如 Recall@K），难以简单并行化单个样本；
    # 若 Loader 内部实现了 evaluate_generation 的 LLM Judge 并行，则会自动生效。
    
    # Task: Retrieval
    if args.evaluation_task in ["retrieval", "all"]:
        try:
            print("\n--- Retrieval Metrics ---")
            r_metrics = loader.evaluate_retrieval()
            print(json.dumps(r_metrics, indent=2))
            final_metrics.update(r_metrics)
        except Exception as e:
            print(f"⚠️ Retrieval evaluation failed: {e}")

    # Task: Generation
    if args.evaluation_task in ["generation", "all"]:
        has_answers = any("final_answer" in s.extra_info for s in loader.samples if str(s.qid) in results_map)
        if has_answers:
            try:
                print("\n--- Generation Metrics ---")
                g_metrics = loader.evaluate_generation()
                print(json.dumps(g_metrics, indent=2))
                final_metrics.update(g_metrics)
            except Exception as e:
                print(f"⚠️ Generation evaluation failed: {e}")
        else:
            if args.evaluation_task == "generation":
                print("⚠️ Warning: No generation answers found in input file. Skipping generation eval.")

    # 4. 保存评估报告
    output_path = os.path.join(args.output_dir, f"evaluation_metrics_{args.evaluation_task}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\n💾 All metrics saved to {output_path}")

if __name__ == "__main__":
    main()