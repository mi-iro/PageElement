python -m vllm.entrypoints.openai.api_server \
  --model /mnt/shared-storage-user/mineru2-shared/madongsheng/saves/Qwen3-VL-8B-Instruct/full/demond_0203_f_base/checkpoint-400 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.7 \
  --max-num-batched-tokens 32768 \
  --mm-processor-cache-gb 0 \
  --compilation_config.cudagraph_mode PIECEWISE