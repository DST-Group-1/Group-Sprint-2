#!/bin/bash
# One-off setup, run on the LOGIN node (it needs the internet, which compute nodes may not have).
# Takes about half an hour; run it inside tmux so a dropped connection does not kill it.
#
#   bash LukeD/bluepebble/setup_login_node.sh            # everything
#   bash LukeD/bluepebble/setup_login_node.sh envs       # or just one part: envs | data | models
#
# Safe to rerun: every part skips what already exists.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=config.sh
source "$HERE/config.sh"
PARTS="${*:-envs data models}"
want() { [[ " $PARTS " == *" $1 "* ]]; }

if [[ "$HERE" != "$BP_DIR" ]]; then
    echo "This folder is $HERE, but config.sh expects the repo at $REPO."
    echo "Clone the repo there (README step 2), or set CASMI_WORK before running."
    exit 1
fi
mkdir -p "$MODELS_DIR" "$CHECKPOINT_DIR" "$OUTPUT_DIR_LARGE" "$LOG_DIR" "$KAGGLEHUB_CACHE"

# ---- Python environments ------------------------------------------------------------------------
# Three, because the strands pin incompatible versions: DreaMS wants Python 3.11 and torch 2.2.1,
# MS2DeepScore pins numpy 2.2.6, and notebook 01 recorded Python 3.14 / polars 2.1.0 / numpy 2.5.4
# in its manifest (same versions here, so the cleaned data comes out identical).
if want envs; then
    if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
        echo "== installing Miniforge into $CONDA_ROOT"
        curl -fsSL -o /tmp/miniforge.sh \
            "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"
        rm /tmp/miniforge.sh
    fi
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    have_env() { conda env list | awk '{print $1}' | grep -qx "$1"; }
    # An env counts as built only if the packages installed last in it import, so one left
    # half-built by a failed pip install is removed and rebuilt on the next run.
    env_ready() {
        have_env "$1" || return 1
        conda run -n "$1" python -c "import $2" >/dev/null 2>&1 && return 0
        echo "== env $1 exists but 'import $2' fails (an earlier install stopped part way): rebuilding it"
        conda remove -y -n "$1" --all
        return 1
    }

    if ! env_ready casmi "polars, torch, pyspark, psutil"; then
        echo "== env casmi: LukeD/01, LukeD/02, ThomasF/03a, Michael's split notebooks"
        conda create -y -n casmi python=3.14 openjdk=21    # Java 21 for PySpark (LukeD/02, section 1.2)
        conda run -n casmi pip install -r "$REPO/requirements.txt" torch pyspark joblib pandas psutil
    fi
    if ! env_ready dreams "dreams, pyopenms, nbconvert"; then
        echo "== env dreams: MichaelM/DreaMS_tuned_scaled_executed (Python 3.11.0, as the DreaMS docs and Michael's run)"
        conda create -y -n dreams python=3.11.0
        [[ -d "$CASMI_WORK/DreaMS" ]] || git clone https://github.com/pluskal-lab/DreaMS.git "$CASMI_WORK/DreaMS"
        # DreaMS pins pyopenms==3.4.0 on Linux, but 3.4.0 only ships manylinux_2_34 wheels (glibc >= 2.34)
        # and BluePebble's glibc is older, so pip finds no build. 3.3.0 (manylinux_2_28) is the version
        # DreaMS itself pins on macOS, with the same numpy requirement; pyopenms is only used to read
        # mzML files (dreams/utils/io.py, lcms.py), not in training or embedding.
        sed -i 's/"pyopenms==3\.4\.0"/"pyopenms==3.3.0"/' "$CASMI_WORK/DreaMS/setup.py"
        conda run -n dreams pip install -e "$CASMI_WORK/DreaMS"
        # Michael's results were made with torch 2.2.1+cu121 (his metrics.json); this build also runs on
        # every BluePebble card type, V100 included.
        conda run -n dreams pip install "torch==2.2.1" --index-url https://download.pytorch.org/whl/cu121
        conda run -n dreams pip install scikit-learn pandas pyarrow matplotlib ipykernel nbconvert
    fi
    if ! env_ready ms2ds "ms2deepscore, matchms, nbconvert"; then
        echo "== env ms2ds: DavidA/MS2DeepScore (the versions pinned in its first cells)"
        conda create -y -n ms2ds python=3.12
        conda run -n ms2ds pip install "numpy==2.2.6" "matchms==0.30.2" "ms2deepscore==2.5.5" \
            pandas pyarrow matplotlib ipywidgets ipykernel nbconvert
    fi
fi

# ---- The data: downloaded from Kaggle, never copied from a laptop --------------------------------
if want data; then
    if [[ ! -f "$RAW_DIR/train.parquet" ]]; then
        if [[ ! -f "$HOME/.kaggle/kaggle.json" && ! -f "$HOME/.kaggle/access_token" && -z "${KAGGLE_KEY:-}${KAGGLE_API_TOKEN:-}" ]]; then
            echo "No Kaggle API token found. Put yours in ~/.kaggle/ (chmod 600) and accept the competition"
            echo "rules on kaggle.com, then rerun: bash $0 data"
            exit 1
        fi
        echo "== downloading the competition files with kagglehub (cache: $KAGGLEHUB_CACHE)"
        path="$("$CONDA_ROOT/bin/conda" run -n casmi python -c \
            "import kagglehub; print(kagglehub.competition_download('$COMPETITION'))" | tail -n 1)"
        ln -sfn "$path" "$RAW_DIR"
    fi
    size=$(stat -L -c %s "$RAW_DIR/train.parquet")
    [[ "$size" == 3033286496 ]] && echo "== train.parquet: $size bytes, the same file as notebook 01's manifest" \
        || echo "!! train.parquet is $size bytes, not the 3,033,286,496 in notebook 01's manifest: Kaggle may have updated it"
fi

# ---- Pretrained weights --------------------------------------------------------------------------
if want models; then
    ckpt="$MODELS_DIR/ssl_model.ckpt"
    if [[ ! -f "$ckpt" ]]; then
        echo "== downloading DreaMS ssl_model.ckpt (1.4 GB) from Zenodo record 10997887"
        curl -fL -o "$ckpt" "https://zenodo.org/records/10997887/files/ssl_model.ckpt?download=1"
    fi
    echo "4deee91dd4486c2b6cbcde1f200b8f50  $ckpt" | md5sum -c -
    if [[ ! -f "$MODELS_DIR/ms2deepscore_model.pt" ]]; then
        echo "!! $MODELS_DIR/ms2deepscore_model.pt is missing: copy in the pretrained MS2DeepScore model"
        echo "   David's notebook uses (ask him which file and where it came from)."
    fi
fi

echo "== setup done. Next: edit config.sh (HPC_ACCOUNT etc.), then run: bash submit_all.sh"
