# python -m vllm.entrypoints.openai.api_server \
#   --served-model-name Qwen3-VL-4B-Instruct \
#   --model /mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/Qwen3-VL-4B-Instruct/ \
#   --port 8000 \
#   --tensor-parallel-size 1 \
#   --gpu-memory-utilization 0.7

export CUDA_VISIBLE_DEVICES=2,3
python -m vllm.entrypoints.openai.api_server \
  --served-model-name Qwen2.5-VL-72B-Instruct \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/modelscope/Qwen/Qwen2___5-VL-72B-Instruct \
  --port 8002 \
  --max_model_len 32768 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9