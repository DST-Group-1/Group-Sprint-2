#!/bin/bash
#SBATCH --job-name=casmi-00-smoke
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Checks, in a few minutes, that every environment sees the GPU and can run on its card type,
# before the long jobs queue for hours. The GPU jobs in submit_all.sh only start if this passes.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware

for env in casmi dreams ms2ds; do
    use_env "$env"
    python - <<'EOF'
import time
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) on {name}, compute capability {cap[0]}.{cap[1]}")
x = torch.randn(4096, 4096, device="cuda", requires_grad=True)
(x @ x).sum().backward()          # fails with "no kernel image" if this torch build has dropped this card
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(20):
    y = x.detach() @ x.detach()
torch.cuda.synchronize()
print(f"fp32 matmul: {20 * 2 * 4096 ** 3 / (time.perf_counter() - start) / 1e12:.1f} TFLOP/s")
EOF
done

use_env casmi
java -version 2>&1 | head -n 1
python -c "import polars, pyspark, rdkit, joblib; print('polars', polars.__version__, '| pyspark', pyspark.__version__, '| rdkit', rdkit.__version__)"
use_env dreams
python -c "from dreams.api import PreTrainedModel; print('dreams imports')"
use_env ms2ds
python -c "import matchms, ms2deepscore; print('matchms', matchms.__version__, '| ms2deepscore', ms2deepscore.__version__)"
for f in "$MODELS_DIR/ssl_model.ckpt" "$MODELS_DIR/ms2deepscore_model.pt"; do
    [[ -f "$f" ]] || echo "!! missing $f (README step 4): the job that needs it will fail"
done
echo "== smoke test passed"
