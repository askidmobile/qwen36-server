#!/bin/bash
set -uo pipefail
export PATH=/workspace/owui-venv/bin:$PATH
export DATA_DIR=/workspace/owui-data
export OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local-ds41
export ENABLE_OPENAI_API=true
export ENABLE_OLLAMA_API=false
export WEBUI_AUTH=false
export ANONYMIZED_TELEMETRY=false
export DO_NOT_TRACK=true
export SCARF_NO_ANALYTICS=true
# RAG включён: эмбеддинги локальные, интернет на сервере есть
export RAG_EMBEDDING_ENGINE=sentence-transformers
export RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export RAG_EMBEDDING_AUTO_UPDATE=false
export ENABLE_RAG_WEB_SEARCH=false
# Держим обе модели в списке: chat без рассуждений, reasoner с ними
export DEFAULT_MODELS=deepseek-v4.1-flash-1017-uncensored-vq
if [ ! -f /workspace/owui-data/.secret ]; then
  head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 43 > /workspace/owui-data/.secret
fi
export WEBUI_SECRET_KEY=$(cat /workspace/owui-data/.secret)
mkdir -p "$DATA_DIR"
exec /workspace/owui-venv/bin/open-webui serve --host 127.0.0.1 --port 8080
