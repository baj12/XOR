#!/bin/bash

# Configuration
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$PROJECT_DIR/logs"
RESOURCE_DIR="$PROJECT_DIR/logs/resources"
PROGRESS_FILE="$PROJECT_DIR/experiment_progress.txt"
FAILED_FILE="$PROJECT_DIR/failed_experiments.txt"
CONFIG_DIR="$PROJECT_DIR/config/yaml"
PYTHON_ENV="xorProject-test"

# Usage information
function show_usage {
    echo "Usage: ./experiment_runner.sh [JOBS] [PATTERN]"
    echo "  JOBS    - Number of parallel jobs (default: 4)"
    echo "  PATTERN - Only run config files matching this pattern (default: all)"
    echo ""
    echo "Examples:"
    echo "  ./experiment_runner.sh                      # Run all configs with 4 parallel jobs"
    echo "  ./experiment_runner.sh 8                    # Run all configs with 8 parallel jobs"
    echo "  ./experiment_runner.sh 4 random_100         # Run only configs containing 'random_100'"
    echo "  ./experiment_runner.sh 4 \"h2_n16\"          # Run only configs with 2 hidden layers and 16 neurons"
    echo "  ./experiment_runner.sh 4 \"random_[15]00\"    # Run configs with 100 or 500 data points"
    exit 1
}

# Check for help argument
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
fi

# Default number of parallel jobs
N_PARALLEL=${1:-4}

# Check if second argument is a number; if not, it's a pattern
if [[ $2 =~ ^[0-9]+$ ]]; then
    N_PARALLEL=$2
    PATTERN=${3:-""}
else
    PATTERN=${2:-""}
fi

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
    --skip-if-exists \\
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

# Find configuration files matching the pattern
if [ -z "$PATTERN" ]; then
    echo "Running ALL configuration files"
    CONFIG_FILES=$(find "$CONFIG_DIR" -name "*.yaml")
else
    echo "Running configuration files matching pattern: $PATTERN"
    CONFIG_FILES=$(find "$CONFIG_DIR" -name "*.yaml" | grep -E "$PATTERN")
fi

# Count matching files
CONFIG_COUNT=$(echo "$CONFIG_FILES" | wc -l)
echo "Found $CONFIG_COUNT matching configuration files:"

# Display the matched files
echo "$CONFIG_FILES" | while read -r file; do
    echo "  $file"
done

# Confirm before running
echo ""
echo "Will run $CONFIG_COUNT experiments with $N_PARALLEL parallel jobs."
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    rm run_experiment.sh
    exit 0
fi

# Run experiments in parallel
echo "Starting parallel execution with $N_PARALLEL jobs..."
echo "$CONFIG_FILES" | \
    parallel --jobs "$N_PARALLEL" \
             --progress \
             --joblog "$PROJECT_DIR/parallel_joblog.txt" \
             ./run_experiment.sh {}

# Clean up
rm run_experiment.sh

echo "All experiments completed"
