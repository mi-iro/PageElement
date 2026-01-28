export CUDA_VISIBLE_DEVICES=1
vllm serve /mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Reranker-8B \
    --served-model-name Qwen3-VL-Reranker-8B \
    --host 0.0.0.0 \
    --port 8004 \
    --runner pooling \
    --max-model-len 8192 \
    --gpu_memory_utilization 0.6 \
    --hf_overrides '{"architectures": ["Qwen3VLForSequenceClassification"],"classifier_from_token": ["no", "yes"],"is_original_qwen3_reranker": true}' \
    --chat-template /mnt/shared-storage-user/mineru3-share/wangzhengren/PageElement/scripts/qwen3_vl_reranker.jinja \
    --allowed-local-media-path /mnt/shared-storage-user/mineru3-share/


# (APIServer pid=2148152) INFO 01-28 19:17:54 [api_server.py:1346] Starting vLLM API server 0 on http://0.0.0.0:8004
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:38] Available routes are:
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /openapi.json, Methods: HEAD, GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /docs, Methods: HEAD, GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /docs/oauth2-redirect, Methods: HEAD, GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /redoc, Methods: HEAD, GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /scale_elastic_ep, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /is_scaling_elastic_ep, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /tokenize, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /detokenize, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /inference/v1/generate, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /pause, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /resume, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /is_paused, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /metrics, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /health, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /load, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/models, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /version, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/responses, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/responses/{response_id}, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/responses/{response_id}/cancel, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/messages, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/chat/completions, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/completions, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/audio/transcriptions, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/audio/translations, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /ping, Methods: GET
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /ping, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /invocations, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /classify, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/embeddings, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /score, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/score, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /rerank, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v1/rerank, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /v2/rerank, Methods: POST
# (APIServer pid=2148152) INFO 01-28 19:17:54 [launcher.py:46] Route: /pooling, Methods: POST
# (APIServer pid=2148152) INFO:     Started server process [2148152]
# (APIServer pid=2148152) INFO:     Waiting for application startup.
# (APIServer pid=2148152) INFO:     Application startup complete.