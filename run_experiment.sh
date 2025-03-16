#!/bin/bash
PROJECT_DIR="/Users/bernd/python/XOR"
config_file=$1
config_name=$(basename "$config_file" .yaml)
timestamp=$(date '+%Y%m%d_%H%M%S')
log_file="$PROJECT_DIR/logs/${config_name}_${timestamp}.log"
resource_log="$PROJECT_DIR/logs/resources/${config_name}_${timestamp}_resources.csv"

echo "Starting experiment: $config_file at $(date)"
echo "Log file: $log_file"

# Change to project directory
cd "$PROJECT_DIR"

# Run the experiment
conda run -n xorProject-test python src/main.py \
    --config "$config_file" \
    --log DEBUG \
    --skip-if-exists \
    2>&1 | tee "$log_file"

status=${PIPESTATUS[0]}

if [ $status -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') SUCCESS $config_file" >> "$PROJECT_DIR/experiment_progress.txt"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED $config_file" >> "$PROJECT_DIR/failed_experiments.txt"
fi
