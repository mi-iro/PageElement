python -m vllm.entrypoints.openai.api_server \
  --model xxx \
  --port 8000 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.7