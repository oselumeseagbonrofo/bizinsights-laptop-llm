#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Launch Multi-Domain Training Pipeline on NVIDIA A6000
# ==============================================================================
set -euo pipefail

ENV_NAME="bizinsight"
CONDA_DIR="$HOME/miniconda3"

echo "============================================================"
echo " Starting Multi-Domain Fine-Tuning & imatrix Quantization"
echo "============================================================"

# 1. Load Conda Environment
if [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
else
    source ~/.bashrc
fi

conda activate "$ENV_NAME"
echo "[OK] Activated Conda environment: $CONDA_DEFAULT_ENV"
echo "[OK] Python: $(which python)"

# 2. Verify GPU
nvidia-smi

# 3. Environment Variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4. Launch Training in unbuffered mode
echo "Launching train_multidomain_a6000.py..."
python -u training_notebooks/train_multidomain_a6000.py 2>&1 | tee training_run.log

echo "============================================================"
echo " Pipeline Complete! Check training_run.log and WandB."
echo "============================================================"
