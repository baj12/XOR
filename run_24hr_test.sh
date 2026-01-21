#!/bin/bash
# 24-Hour Continuous Learning Test Run
# Started: $(date)

# Activate conda environment
source /Users/bernd/miniconda3/bin/activate xorProject

# Set working directory
cd /Users/bernd/python/XOR

# Set log file
LOG_DIR="logs/continuous_test_24hr"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

echo "==================================================="
echo "Starting 24-Hour Continuous Learning Test"
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "Database: data/continuous/test_24hr.db"
echo "Models: models/continuous_test_24hr/"
echo "Reports: reports/continuous_test_24hr/"
echo "==================================================="

# Run orchestrator for 24 hours
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/test_24hr.db \
    --model-dir models/continuous_test_24hr \
    --report-dir reports/continuous_test_24hr \
    --test-hours 24 \
    --log INFO 2>&1 | tee "$LOG_FILE"

# Summary
echo ""
echo "==================================================="
echo "Test Run Complete"
echo "End time: $(date)"
echo "Log file: $LOG_FILE"
echo "==================================================="
