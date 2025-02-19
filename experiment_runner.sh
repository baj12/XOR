#!/bin/bash

# Configuration
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$PROJECT_DIR/logs"
RESOURCE_DIR="$PROJECT_DIR/logs/resources"
PROGRESS_FILE="$PROJECT_DIR/experiment_progress.txt"
FAILED_FILE="$PROJECT_DIR/failed_experiments.txt"
CONFIG_DIR="$PROJECT_DIR/config/yaml"
PYTHON_ENV="xorProject-test"


# Default number of parallel jobs
N_PARALLEL=${1:-4}

# Create necessary directories
mkdir -p "$LOG_DIR" "$RESOURCE_DIR"
touch "$PROGRESS_FILE"
touch "$FAILED_FILE"

# Handle GNU Parallel citation
if [ ! -f ~/.parallel/will-cite ]; then
    mkdir -p ~/.parallel
    touch ~/.parallel/will-cite
fi

# Create a temporary script for the single experiment
cat > run_experiment.sh << EOF
#!/bin/bash
PROJECT_DIR="$PROJECT_DIR"
config_file=\$1
config_name=\$(basename "\$config_file" .yaml)
timestamp=\$(date '+%Y%m%d_%H%M%S')
log_file="\$PROJECT_DIR/logs/\${config_name}_\${timestamp}.log"
resource_log="\$PROJECT_DIR/logs/resources/\${config_name}_\${timestamp}_resources.csv"

echo "Starting experiment: \$config_file at \$(date)"
echo "Log file: \$log_file"

# Change to project directory
cd "\$PROJECT_DIR"

# Run the experiment
conda run -n xorProject-test python src/main.py \\
    --config "\$config_file" \\
    --log DEBUG \\
    2>&1 | tee "\$log_file"

status=\${PIPESTATUS[0]}

if [ \$status -eq 0 ]; then
    echo "\$(date '+%Y-%m-%d %H:%M:%S') SUCCESS \$config_file" >> "\$PROJECT_DIR/experiment_progress.txt"
else
    echo "\$(date '+%Y-%m-%d %H:%M:%S') FAILED \$config_file" >> "\$PROJECT_DIR/failed_experiments.txt"
fi
EOF

# Make the script executable
chmod +x run_experiment.sh

# Print found configuration files
echo "Found configuration files:"
find "$CONFIG_DIR" -name "*.yaml" | while read -r file; do
    echo "  $file"
done

# Run experiments in parallel
echo "Starting parallel execution with $N_PARALLEL jobs..."
find "$CONFIG_DIR" -name "*.yaml" | \
    parallel --jobs "$N_PARALLEL" \
             --progress \
             --joblog "$PROJECT_DIR/parallel_joblog.txt" \
             ./run_experiment.sh {}

# Clean up
rm run_experiment.sh

echo "All experiments completed"
