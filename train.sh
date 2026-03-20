#!/bin/bash
###############################################################################
# Submit SLEAP training and monitor automatically
#
# Usage:
#   bash train.sh           # 50 epochs (default)
#   bash train.sh 20        # 20 epochs
###############################################################################

EPOCHS=${1:-50}

echo "Submitting ${EPOCHS}-epoch training job..."
JOBID=$(sbatch --parsable submit_bunya.sh "$EPOCHS")

if [ -z "$JOBID" ]; then
    echo "ERROR: sbatch failed. Check submit_bunya.sh"
    exit 1
fi

echo "Job submitted: $JOBID"
echo ""

# Launch monitor with the new job ID
bash monitor.sh "$JOBID"
