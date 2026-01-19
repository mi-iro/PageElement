python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/demond_vidore_5k_0114_noresize/checkpoint-300 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7