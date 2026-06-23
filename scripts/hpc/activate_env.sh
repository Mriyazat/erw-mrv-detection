
module load gcc arrow/16.1.0 cuda 2>/dev/null || module load gcc arrow cuda

source "${HOME}/erw_venv/bin/activate"

# offline HuggingFace cache for the foundation models (compute nodes have no net)
export HF_HOME="${HOME}/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -c "import pyarrow, pandas, numpy; \
print('env ok | pyarrow', pyarrow.__version__, '| numpy', numpy.__version__)" \
  || echo "WARNING: pyarrow/pandas not importable - check module order"
