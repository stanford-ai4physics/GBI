#!/bin/bash

# Activation script for GBI environment on M2 Mac
# Usage: source activate_gbi.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Initialize conda
eval "$(/opt/homebrew/Caskroom/mambaforge/base/bin/conda shell.bash hook)"

# Activate the local conda environment
conda activate "${SCRIPT_DIR}/.conda-envs/gbi_ranode"

# Set up project paths (from main setup.sh)
export PATH="${SCRIPT_DIR}/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}/ranode:${SCRIPT_DIR}:${PYTHONPATH}"

# Set up LAW and data directories (ranode-specific settings)
export LAW_HOME="${SCRIPT_DIR}/ranode/.law"
export LAW_CONFIG_FILE="${SCRIPT_DIR}/ranode/law.cfg"

# Load data/output directory configuration
if [[ -f "${SCRIPT_DIR}/ranode/.config" ]]; then
    source "${SCRIPT_DIR}/ranode/.config"
fi

echo ""
echo "✓ GBI environment activated!"
echo "  Python: $(python --version)"
echo "  Environment: ${CONDA_PREFIX}"
echo "  Working directory: ${SCRIPT_DIR}"
