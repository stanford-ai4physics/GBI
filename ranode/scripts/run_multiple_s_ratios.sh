#!/bin/bash
# Run pipeline with multiple s_ratio_index values for linearity check

# Activate environment
cd /Users/victorzhang/GBI
source activate_gbi.sh
cd ranode

# Array of s_ratio_index values to test
# Using fewer points for faster testing: 5, 8, 11 (instead of all 0-12)
S_RATIO_INDICES=(5 8 11)

VERSION="linearity_test"

echo "=========================================="
echo "Running linearity check with s_ratio_index: ${S_RATIO_INDICES[@]}"
echo "=========================================="

for S_IDX in "${S_RATIO_INDICES[@]}"; do
    echo ""
    echo "===================="
    echo "Running s_ratio_index = $S_IDX"
    echo "===================="

    law run FittingScanResults \
        --version $VERSION \
        --ensemble 1 \
        --mx 100 \
        --my 500 \
        --s-ratio-index $S_IDX \
        --workers 1 \
        --FittingScanResults-device mps \
        --BkgTemplateTraining-device mps \
        --BkgTemplateChecking-device mps \
        --PerfectBkgTemplateTraining-device mps \
        --RNodeTemplate-device mps \
        --PredictBkgProb-device mps \
        --ScanRANODE-device mps \
        --SampleModelBinSR-device mps \
        --PredictBkgProbGen-device mps

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for s_ratio_index=$S_IDX"
    else
        echo "SUCCESS: Completed s_ratio_index=$S_IDX"
    fi
done

echo ""
echo "=========================================="
echo "All runs completed!"
echo "Now run: python scripts/generate_fig4_linearity_check.py"
echo "=========================================="
