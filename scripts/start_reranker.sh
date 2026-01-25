export CUDA_VISIBLE_DEVICES=3
python -m vllm.entrypoints.openai.api_server \
    --served-model-name Qwen3-VL-Reranker-8B \
    --model /mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B \
    --trust-remote-code \
    --gpu-memory-utilization 0.75 \
    --max-model-len 16384 \
    --port 8002