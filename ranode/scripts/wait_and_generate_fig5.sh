#!/bin/bash
# Wait for test_full to complete and automatically generate Fig 5

cd /Users/victorzhang/GBI
source activate_gbi.sh
cd ranode

echo "=========================================="
echo "Waiting for test_full to complete..."
echo "=========================================="

# Check if test_full has any trained models
while true; do
    MODEL_COUNT=$(find /Users/victorzhang/GBI/output/version_test_full/RNodeTemplate -name "model_S.pt" 2>/dev/null | wc -l)
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Found $MODEL_COUNT trained models"

    # We expect 20 (scan points) x 20 (templates) = 400 models
    # But let's wait for at least 200 to be safe
    if [ "$MODEL_COUNT" -ge 200 ]; then
        echo "Sufficient models found! Generating Fig 5..."
        python scripts/generate_fig5_distribution_comparison.py

        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Fig 5 generated successfully!"
            echo "📊 Location: /Users/victorzhang/GBI/output/version_test_full/fig5_distribution_comparison.pdf"
            break
        else
            echo "❌ Fig 5 generation failed. Will retry in 5 minutes..."
        fi
    fi

    # Wait 5 minutes before checking again
    sleep 300
done

echo ""
echo "Done! You can now view the Fig 5 plot."
echo "Open with: open /Users/victorzhang/GBI/output/version_test_full/fig5_distribution_comparison.pdf"
