#!/bin/bash
###############################################################################
# Monitor SLEAP training jobs on Bunya
#
# Usage:
#   bash monitor.sh                  # auto-detect latest job
#   bash monitor.sh 22584516         # specific job ID
###############################################################################

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Get job ID
if [ -n "$1" ]; then
    JOBID="$1"
else
    # Auto-detect: find the most recent sleap log
    LOGFILE=$(ls -t sleap_*.log 2>/dev/null | head -1)
    if [ -z "$LOGFILE" ]; then
        echo -e "${RED}No sleap log files found. Submit a job first:${NC}"
        echo "  sbatch submit_bunya.sh 20"
        exit 1
    fi
    JOBID=$(echo "$LOGFILE" | grep -oP '\d+')
fi

LOGFILE="sleap_${JOBID}.log"

if [ ! -f "$LOGFILE" ]; then
    echo -e "${YELLOW}Waiting for log file: ${LOGFILE}${NC}"
    echo "(Job may be queued. Check with: squeue -u \$USER)"
    while [ ! -f "$LOGFILE" ]; do
        sleep 2
    done
fi

# Extract epochs from the job's config line in the log
get_total_epochs() {
    grep -oP 'Epochs:\s+\K\d+' "$LOGFILE" 2>/dev/null || echo "0"
}

# Draw progress bar
draw_bar() {
    local current=$1
    local total=$2
    local width=40

    if [ "$total" -eq 0 ]; then
        return
    fi

    local pct=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))

    local bar=""
    for ((i = 0; i < filled; i++)); do bar+="█"; done
    for ((i = 0; i < empty; i++)); do bar+="░"; done

    echo -ne "\r  ${CYAN}[${bar}]${NC} ${BOLD}${pct}%${NC} (${current}/${total} epochs)"
}

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         SLEAP Training Monitor - Job ${JOBID}  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# Track what stage we're in
last_epoch=0
total_epochs=0
stage="starting"
last_line_count=0

while true; do
    # Check if job is still running
    job_state=$(squeue -j "$JOBID" -h -o "%T" 2>/dev/null)

    if [ -z "$job_state" ] && [ "$stage" != "done" ]; then
        # Job finished or was cancelled - check log for final status
        if grep -q "Done! Results are in:" "$LOGFILE" 2>/dev/null; then
            stage="done"
        elif grep -q "Error\|error\|FAILED\|Traceback" "$LOGFILE" 2>/dev/null; then
            stage="error"
        else
            stage="done"
        fi
    fi

    # Detect stage from log content
    if grep -q "No such file or directory\|ModuleNotFoundError\|Traceback\|FAILED" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" != "error_shown" ]; then
            echo ""
            echo -e "${RED}✗ ERROR DETECTED${NC}"
            echo -e "${RED}─────────────────────────────────────────${NC}"
            tail -5 "$LOGFILE"
            echo -e "${RED}─────────────────────────────────────────${NC}"
            echo -e "Full log: ${YELLOW}cat ${LOGFILE}${NC}"
            stage="error_shown"
        fi
        break
    fi

    if [ "$stage" = "done" ]; then
        echo ""
        echo -e "${GREEN}✓ TRAINING COMPLETE${NC}"
        output_dir=$(grep -oP "Done! Results are in: \K.*" "$LOGFILE" 2>/dev/null)
        if [ -n "$output_dir" ]; then
            echo -e "  Results: ${CYAN}${output_dir}/${NC}"
            if [ -d "$output_dir" ]; then
                echo -e "  Files:   $(ls "$output_dir" 2>/dev/null | tr '\n' '  ')"
            fi
        fi
        break
    fi

    # Get total epochs if not yet found
    if [ "$total_epochs" -eq 0 ]; then
        total_epochs=$(get_total_epochs)
    fi

    # Detect current stage
    if grep -q "downloading\|Downloading\|Dataset downloaded" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" != "download" ]; then
            stage="download"
            echo -e "${YELLOW}⟳ Downloading dataset from Roboflow...${NC}"
        fi
    fi

    if grep -q "Converting\|Saved:.*labeled frames\|Skeleton saved" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" != "convert" ]; then
            stage="convert"
            echo -e "${GREEN}✓ Dataset downloaded${NC}"
            echo -e "${YELLOW}⟳ Converting COCO → SLEAP format...${NC}"
        fi
    fi

    if grep -q "Dataset already exists" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" != "skip_data" ]; then
            stage="skip_data"
            echo -e "${GREEN}✓ Dataset already exists, skipping download${NC}"
        fi
    fi

    if grep -q "Starting training\|Epoch\|epoch" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" != "training" ]; then
            stage="training"
            echo -e "${YELLOW}⟳ Training in progress...${NC}"
        fi

        # Parse epoch progress - try common formats
        current_epoch=$(grep -oP '[Ee]poch\s*\K\d+' "$LOGFILE" 2>/dev/null | tail -1)
        if [ -z "$current_epoch" ]; then
            current_epoch=$(grep -oP '\d+/\d+' "$LOGFILE" 2>/dev/null | tail -1 | cut -d'/' -f1)
        fi

        if [ -n "$current_epoch" ] && [ "$total_epochs" -gt 0 ]; then
            if [ "$current_epoch" -ne "$last_epoch" ]; then
                last_epoch=$current_epoch
                draw_bar "$current_epoch" "$total_epochs"
            fi
        fi
    fi

    if grep -q "Training complete\|Model copied" "$LOGFILE" 2>/dev/null; then
        if [ "$stage" = "training" ] && [ "$total_epochs" -gt 0 ]; then
            draw_bar "$total_epochs" "$total_epochs"
            echo ""
        fi
        echo -e "${GREEN}✓ Training finished${NC}"
        echo -e "${YELLOW}⟳ Collecting results...${NC}"
        stage="collecting"
    fi

    # Show job queue status
    if [ -n "$job_state" ]; then
        case "$job_state" in
            PENDING)
                if [ "$stage" = "starting" ]; then
                    echo -ne "\r  ${YELLOW}⏳ Job queued, waiting for GPU...${NC}  "
                fi
                ;;
        esac
    fi

    sleep 3
done

echo ""
