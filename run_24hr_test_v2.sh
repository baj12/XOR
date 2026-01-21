#!/bin/bash
# 24-Hour Continuous Learning Test Run (v2 - with relaxed duration requirements)
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
echo "Starting 24-Hour Continuous Learning Test (v2)"
echo "Start time: $(date)"
echo "Config: continuous_learning_test_24hr.yaml"
echo "  - Min duration: 10 seconds (relaxed for testing)"
echo "  - Min samples for training: 100 (relaxed)"
echo "  - Device validation: disabled"
echo "Log file: $LOG_FILE"
echo "Database: data/continuous/test_24hr.db"
echo "==================================================="

# Run orchestrator for 24 hours with modified config
python -m src.continuous.orchestrator \
    --config config/continuous_learning_test_24hr.yaml \
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
