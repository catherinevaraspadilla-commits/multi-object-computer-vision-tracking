#!/bin/bash
# Submit both 20-epoch and 50-epoch training jobs to Bunya
# Usage: bash run_both.sh

echo "Submitting 20-epoch job..."
JOB1=$(sbatch --parsable submit_bunya.sh 20)
echo "  Job ID: $JOB1"

echo "Submitting 50-epoch job..."
JOB2=$(sbatch --parsable submit_bunya.sh 50)
echo "  Job ID: $JOB2"

echo ""
echo "Both jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Results will be saved to:"
echo "  results_20ep/"
echo "  results_50ep/"
