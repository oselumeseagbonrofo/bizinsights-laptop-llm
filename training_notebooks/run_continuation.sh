#!/usr/bin/env bash
# ==============================================================================
# Launcher for Post-Training Continuation Pipeline on NVIDIA A6000
# ==============================================================================
set -euo pipefail

ENV_NAME="bizinsight"
CONDA_DIR="$HOME/miniconda3"

echo "============================================================"
echo " Starting Post-Training Continuation (Export & Quantize)"
echo "============================================================"

# 1. Load Conda Environment
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    source ~/.bashrc
fi

conda activate "$ENV_NAME"
echo "[OK] Activated Conda environment: $CONDA_DEFAULT_ENV"

# 2. Permissions fix on HuggingFace cache and model_f16
echo "Ensuring full write permissions..."
chmod -R u+w ~/.cache/huggingface/ 2>/dev/null || true
chmod -R u+w ~/bizinsight/model_f16/ 2>/dev/null || true

# 3. Environment Variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="/home/shadeform/.unsloth/llama.cpp:${LD_LIBRARY_PATH:-}"

# 4. Run Continuation Script
cd ~/bizinsight
echo "Launching continue_pipeline_a6000.py..."
python -u training_notebooks/continue_pipeline_a6000.py 2>&1 | tee continuation_run.log

# Also append to main training_run.log
echo -e "\n\n=== CONTINUATION LOG APPENDED ===" >> training_run.log
cat continuation_run.log >> training_run.log

echo "============================================================"
echo " Continuation Complete! Models published to Hugging Face."
echo "============================================================"
