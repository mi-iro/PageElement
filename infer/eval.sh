CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

conda activate eval

# pip install matplotlib opencv-python prettytable seaborn numpy tqdm

# 并行处理版本
# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v1.py \
#     --pred_path /mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/inference_cache/qwen3-vl-plus/ViDoRe_Test_Positive_cache.json \
#     --data_path /mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/data_pipeline/vidore/processed_dataset/test_pos.json \
#     --output_dir ./eval_new/ \
#     --workers 16 \
#     --betas 1.0

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#     --pred_path /mnt/shared-storage-user/mineru3-share/jiayu/Agent/eval_results/checkpoint-300/vidore_neg \
#     --data_path /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/vidore/test_neg.json \
#     --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/checkpoint-300/vidore_neg \
#     --workers 16 \
#     --betas 1.0

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#     --pred_path /mnt/shared-storage-user/mineru3-share/jiayu/Agent/eval_results/checkpoint-400/mvtoolbench_full \
#     --data_path /mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark/mvtoolbench_full.json \
#     --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/checkpoint-400/mvtoolbench_full \
#     --workers 16 \
#     --betas 1.0


# /mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark/mvtoolbench_Standard_Document.json
# /mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark/mvtoolbench_full.json

# /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/vidore/test_neg.json
# /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/vidore/test_pos.json


# 训练数据
# /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/train_vidore
# /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/train_codevision

# gt
# /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/train_merged.json
# /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/codevision/codevision_rl_with_ids.json

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#      --pred_path /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/train_vidore \
#      --data_path /mnt/shared-storage-user/mineru2-shared/madongsheng/dataset/train_merged.json \
#      --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/new_eval/page_075_07/eval_train_vidore \
#      --workers 32

# /mnt/shared-storage-user/mineru2-shared/madongsheng/results/0121model

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#     --pred_path /mnt/shared-storage-user/mineru2-shared/madongsheng/results/0121model/mvtbch \
#     --data_path /mnt/shared-storage-user/mineru3-share/jiayu/newBench/dataOri/MVToolBench/mvtoolbench_benchmark/mvtoolbench_full.json \
#     --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/0121model/eval_mvtbch \
#     --workers 32

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#     --pred_path /mnt/shared-storage-user/mineru2-shared/madongsheng/results/0121model/test_neg \
#     --data_path /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/vidore/test_neg.json \
#     --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/0121model/vidore/test_neg \
#     --workers 32

# python /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/eval_v2.py \
#     --pred_path /mnt/shared-storage-user/mineru2-shared/madongsheng/results/0121model/test_pos \
#     --data_path /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/vidore/test_pos.json \
#     --output_dir /mnt/shared-storage-user/mineru3-share/jiayu/evaluation/result/0121model/vidore/test_pos \
#     --workers 32

cd infer
python eval_v2.py \
    --pred_path /mnt/shared-storage-user/madongsheng/Agent_0128/inference/vlplus/finrag_neg \
    --data_path eval_json/neg_500.json \
    --output_dir /mnt/shared-storage-user/madongsheng/Agent_0128/inference/vlplus/finrag_neg_result \
    --workers 32

#finrag
python eval_v2.py \
    --pred_path /mnt/shared-storage-user/madongsheng/Agent_0128/inference/sft400_new_resize/finrag \
    --data_path eval_json/finragbench_bbox_test.json \
    --output_dir /mnt/shared-storage-user/madongsheng/Agent_0128/inference/sft400_new_resize/finrag_result \
    --workers 32

#mvtb
python eval_v2.py \
    --pred_path /mnt/shared-storage-user/madongsheng/Agent_0128/inference/sft400_new_resize/mvtb \
    --data_path eval_json/mvtoolbench_benchmark/mvtoolbench_full.json \
    --output_dir /mnt/shared-storage-user/madongsheng/Agent_0128/inference/sft400_new_resize/mvtb_result \
    --workers 32



#测评gemini的 请忽略
python /mnt/shared-storage-user/madongsheng/Agent_0128/eval/eval_v2_y.py  \
    --pred_path /mnt/shared-storage-user/madongsheng/Agent_0128/vidore_distill/result_process \
    --data_path /mnt/shared-storage-user/mineru2-shared/jiayu/data/vidore/vidore_40_split/data.json \
    --output_dir /mnt/shared-storage-user/madongsheng/Agent_0128/vidore_distill/result_process_result \
    --workers 32




python /mnt/shared-storage-user/madongsheng/Agent_0128/eval/eval_v2_y.py \
    --pred_path /mnt/shared-storage-user/madongsheng/Agent_0128/vidore_distill/result \
    --data_path /mnt/shared-storage-user/mineru2-shared/jiayu/data/vidore/vidore_40_split/data.json \
    --output_dir /mnt/shared-storage-user/madongsheng/Agent_0128/vidore_distill_vision \
    --workers 32
#file_path = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V/data/citation_labels/citation_labels_new/finragbench_bbox_test.json"
