#!/bin/bash

# experiment_runner.sh
set -e  # Exit on error

# Configuration
LOG_DIR="logs"
RESOURCE_DIR="logs/resources"
PROGRESS_FILE="experiment_progress.txt"
FAILED_FILE="failed_experiments.txt"
CONFIG_DIR="config/yaml"
PYTHON_ENV="xorProject-test"

# Cleanup function
cleanup() {
    local pids=$(jobs -p)
    if [ ! -z "$pids" ]; then
        echo "Cleaning up processes..."
        kill $pids 2>/dev/null || true
    fi
}

# Set up trap for cleanup
trap cleanup EXIT INT TERM

# Create necessary directories and files
mkdir -p "$LOG_DIR" "$RESOURCE_DIR"
touch "$PROGRESS_FILE"
touch "$FAILED_FILE"

# Function to check if a process is running
is_process_running() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Function to monitor resources
monitor_resources() {
    local pid=$1
    local resource_log=$2
    
    echo "Timestamp,CPU%,Memory(MB),Virtual Memory(MB),Threads,ProcessCount" > "$resource_log"
    
    while is_process_running $pid; do
        if [ "$(uname)" == "Darwin" ]; then
            # macOS specific commands
            # Get all Python processes that are children of the main process
            local pids=$(pgrep -P $pid python 2>/dev/null)
            pids="$pid $pids"  # Add main process to list
            
            # Initialize counters
            local total_cpu=0
            local total_mem=0
            local total_vmem=0
            local total_threads=0
            
            # Sum up resources for all processes
            for p in $pids; do
                if ps -p $p >/dev/null 2>&1; then
                    # Get CPU and memory stats
                    local cpu=$(ps -p $p -o %cpu= 2>/dev/null || echo "0")
                    local mem=$(ps -p $p -o rss= 2>/dev/null || echo "0")
                    local vmem=$(ps -p $p -o vsz= 2>/dev/null || echo "0")
                    local threads=$(ps -M -p $p 2>/dev/null | grep -v "USER" | wc -l | tr -d ' ')
                    
                    total_cpu=$(echo "$total_cpu + $cpu" | bc)
                    total_mem=$(echo "$total_mem + $mem" | bc)
                    total_vmem=$(echo "$total_vmem + $vmem" | bc)
                    total_threads=$(echo "$total_threads + $threads" | bc)
                fi
            done
            
            # Convert memory from KB to MB
            total_mem=$(echo "scale=2; $total_mem / 1024" | bc)
            total_vmem=$(echo "scale=2; $total_vmem / 1024" | bc)
            
            # Count total Python processes
            local process_count=$(echo "$pids" | wc -w)
            
        else
            # Linux commands
            local pids=$(pgrep -P $pid python 2>/dev/null)
            pids="$pid $pids"
            
            local total_cpu=0
            local total_mem=0
            local total_vmem=0
            local total_threads=0
            
            for p in $pids; do
                if ps -p $p >/dev/null 2>&1; then
                    local cpu=$(ps -p $p -o %cpu= 2>/dev/null || echo "0")
                    local mem=$(ps -p $p -o rss= 2>/dev/null || echo "0")
                    local vmem=$(ps -p $p -o vsz= 2>/dev/null || echo "0")
                    local threads=$(ps -L -p $p | wc -l)
                    
                    total_cpu=$(echo "$total_cpu + $cpu" | bc)
                    total_mem=$(echo "$total_mem + $mem" | bc)
                    total_vmem=$(echo "$total_vmem + $vmem" | bc)
                    total_threads=$(echo "$total_threads + $threads" | bc)
                fi
            done
            
            # Convert memory from KB to MB
            total_mem=$(echo "scale=2; $total_mem / 1024" | bc)
            total_vmem=$(echo "scale=2; $total_vmem / 1024" | bc)
            
            local process_count=$(echo "$pids" | wc -w)
        fi
        
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp,$total_cpu,$total_mem,$total_vmem,$total_threads,$process_count" >> "$resource_log"
        sleep 5
    done
}

# Function to run single experiment
run_experiment() {
    local config_file="$1"
    local config_name=$(basename "$config_file" .yaml)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$LOG_DIR/${config_name}_${timestamp}.log"
    local resource_log="$RESOURCE_DIR/${config_name}_${timestamp}_resources.csv"

    echo "Starting experiment: $config_file at $(date)"
    echo "Log file: $log_file"
    echo "Resource monitoring: $resource_log"

    # Activate conda environment
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        . "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$PYTHON_ENV"
    else
        echo "Error: Conda not found"
        return 1
    fi

    # Start the Python process
    python src/main.py --config "$config_file" --log DEBUG > "$log_file" 2>&1 &
    local python_pid=$!

    # Start resource monitoring
    monitor_resources $python_pid "$resource_log" &
    local monitor_pid=$!

    # Wait for Python process to complete
    wait $python_pid
    local exit_status=$?

    # Stop monitoring
    if is_process_running $monitor_pid; then
        kill $monitor_pid 2>/dev/null || true
        wait $monitor_pid 2>/dev/null || true
    fi

    # Generate plots if the resource log exists and has data
    if [ -f "$resource_log" ] && [ -s "$resource_log" ]; then
        generate_resource_plots "$resource_log"
    fi

    if [ $exit_status -eq 0 ]; then
        echo "$config_file:COMPLETED:$(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
        echo "Successfully completed: $config_file"
        return 0
    else
        local error_msg=$(tail -n 5 "$log_file" | tr '\n' ' ')
        echo "$config_file:FAILED:$error_msg:$(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
        echo "Failed: $config_file"
        return 1
    fi
}

# Function to generate resource plots
generate_resource_plots() {
    local resource_log=$1
    local plot_file="${resource_log%.csv}_plots.png"
    
    python - <<EOF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    # Read the resource log
    df = pd.read_csv('$resource_log')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    plt.suptitle('Resource Usage Over Time\\n$(basename "$resource_log")')

    # CPU Usage
    sns.lineplot(data=df, x='Timestamp', y='CPU%', ax=axs[0,0])
    axs[0,0].set_title('CPU Usage')
    axs[0,0].tick_params(axis='x', rotation=45)

    # Memory Usage
    sns.lineplot(data=df, x='Timestamp', y='Memory(MB)', ax=axs[0,1])
    axs[0,1].set_title('Memory Usage')
    axs[0,1].tick_params(axis='x', rotation=45)

    # Process Count
    sns.lineplot(data=df, x='Timestamp', y='ProcessCount', ax=axs[1,0])
    axs[1,0].set_title('Python Process Count')
    axs[1,0].tick_params(axis='x', rotation=45)

    # Thread Count
    sns.lineplot(data=df, x='Timestamp', y='Threads', ax=axs[1,1])
    axs[1,1].set_title('Thread Count')
    axs[1,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('$plot_file', dpi=300, bbox_inches='tight')
    plt.close()

    # Generate statistics summary
    stats = df.describe()
    stats.to_csv('${resource_log%.csv}_summary.csv')
except Exception as e:
    print(f"Error generating plots: {e}")
EOF
}

# Main execution
echo "Starting experiments at $(date)"
echo "Configuration directory: $CONFIG_DIR"

# Get all yaml files
config_files=$(find "$CONFIG_DIR" -name "*.yaml" | sort)
total_configs=$(echo "$config_files" | wc -l)
current=0
failed_count=0

# Process each config file
for config_file in $config_files; do
    ((current++))
    echo "Processing $current of $total_configs: $config_file"
    
    if grep -q "^$config_file:COMPLETED" "$PROGRESS_FILE"; then
        echo "Skipping completed experiment: $config_file"
        continue
    fi

    if ! run_experiment "$config_file"; then
        ((failed_count++))
        echo "Error in experiment: $config_file"
        if [ $failed_count -ge 3 ]; then
            echo "Too many failures ($failed_count). Stopping execution."
            exit 1
        fi
    fi
done

# Print summary
echo -e "\nExperiment run completed at $(date)"
echo "Summary:"
echo "Total configurations: $total_configs"
echo "Completed experiments: $(grep -c ":COMPLETED" "$PROGRESS_FILE")"
echo "Failed experiments: $(grep -c ":FAILED" "$FAILED_FILE")"

if [ -s "$FAILED_FILE" ]; then
    echo -e "\nFailed experiments:"
    cat "$FAILED_FILE"
fi
