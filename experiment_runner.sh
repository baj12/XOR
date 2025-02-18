#!/bin/bash

## need to modify poweruser permission
# sudo visudo -f /etc/sudoers.d/powermetrics
# yourusername ALL=(root) NOPASSWD: /usr/bin/powermetrics
# sudo chmod 440 /etc/sudoers.d/powermetrics
# sudo powermetrics --show-gpu --show-ane -i 1000 -n 1


# experiment_runner.sh
set -e  # Exit on error

# Function to print messages that works in both bash and zsh
log_message() {
    echo "$@"
}


# Configuration
LOG_DIR="logs"
RESOURCE_DIR="logs/resources"
PROGRESS_FILE="experiment_progress.txt"
FAILED_FILE="failed_experiments.txt"
CONFIG_DIR="config/yaml"
PYTHON_ENV="xorProject"

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
    local detail_log="${resource_log%.*}_details.log"  # Create detail log filename
    
    # Check if we're on Apple Silicon
    local is_apple_silicon=$(uname -p | grep -q "arm" && echo "true" || echo "false")
    
    # Initialize the resource log
    echo "Timestamp,CPU%,Memory(MB),Virtual Memory(MB),Threads,ProcessCount,GPU Memory(MB),GPU Utilization(%),ANE Usage" > "$resource_log"
    
    # Initialize the detail log with timestamp
    echo "Resource Monitoring Details - Started at $(date)" > "$detail_log"
    echo "================================================" >> "$detail_log"
    
    while is_process_running $pid; do
        # Add timestamp to detail log
        echo -e "\nTimestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$detail_log"
        echo "----------------" >> "$detail_log"
        
        # Initialize counters
        local total_cpu=0
        local total_mem=0
        local total_vmem=0
        local total_threads=0
        local process_count=0
        
        # Get all Python processes that are children of the main process
        local pids="$pid $(pgrep -P $pid python 2>/dev/null)"
        echo "Monitoring PIDs: $pids" >> "$detail_log"
        
        for p in $pids; do
            if ps -p $p >/dev/null 2>&1; then
                if [ "$(uname)" == "Darwin" ]; then
                    # Using separate ps commands to ensure reliable parsing
                    local cpu=$(ps -p $p -o %cpu= 2>/dev/null | tr -d ' ')
                    local mem=$(ps -p $p -o rss= 2>/dev/null | tr -d ' ')
                    local vmem=$(ps -p $p -o vsz= 2>/dev/null | tr -d ' ')
                    local threads=$(ps -M -p $p 2>/dev/null | grep -v "USER" | wc -l | tr -d ' ')
                    
                    # Write process details to detail log
                    {
                        echo "Process $p individual values:"
                        echo "  CPU: $cpu"
                        echo "  MEM: $mem"
                        echo "  VMEM: $vmem"
                        echo "  Threads: $threads"
                    } >> "$detail_log"
                    
                    if [ ! -z "$cpu" ] && [ ! -z "$mem" ] && [ ! -z "$vmem" ] && [ ! -z "$threads" ]; then
                        total_cpu=$(echo "$total_cpu + $cpu" | bc 2>/dev/null || echo "$total_cpu")
                        total_mem=$(echo "$total_mem + $mem" | bc 2>/dev/null || echo "$total_mem")
                        total_vmem=$(echo "$total_vmem + $vmem" | bc 2>/dev/null || echo "$total_vmem")
                        total_threads=$(echo "$total_threads + $threads" | bc 2>/dev/null || echo "$total_threads")
                        ((process_count++))
                    fi
                fi
            fi
        done
        
        # Convert memory from KB to MB
        if [ ! -z "$total_mem" ] && [ "$total_mem" != "0" ]; then
            total_mem=$(echo "scale=2; $total_mem / 1024" | bc 2>/dev/null || echo "0")
        fi
        if [ ! -z "$total_vmem" ] && [ "$total_vmem" != "0" ]; then
            total_vmem=$(echo "scale=2; $total_vmem / 1024" | bc 2>/dev/null || echo "0")
        fi
        
        # Write totals to detail log
        {
            echo "Total values:"
            echo "  CPU: $total_cpu"
            echo "  MEM (MB): $total_mem"
            echo "  VMEM (MB): $total_vmem"
            echo "  Threads: $total_threads"
            echo "  Process Count: $process_count"
            echo "----------------------------------------"
        } >> "$detail_log"
        
        # Get GPU metrics for Apple Silicon
        local gpu_memory="0"
        local gpu_util="0"
        local ane_usage="0"
        
        if [ "$is_apple_silicon" = "true" ]; then
            if [ -f "/tmp/powermetrics_$pid.txt" ]; then
                gpu_util=$(grep "GPU Active" "/tmp/powermetrics_$pid.txt" | tail -n 1 | awk '{print $3}' | sed 's/%//' || echo "0")
                gpu_memory=$(grep "GPU Memory" "/tmp/powermetrics_$pid.txt" | tail -n 1 | awk '{print $3}' || echo "0")
                ane_usage=$(grep "ANE Active" "/tmp/powermetrics_$pid.txt" | tail -n 1 | awk '{print $3}' | sed 's/%//' || echo "0")
                
                # Write GPU metrics to detail log
                {
                    echo "GPU Metrics:"
                    echo "  GPU Utilization: $gpu_util%"
                    echo "  GPU Memory: $gpu_memory MB"
                    echo "  ANE Usage: $ane_usage%"
                } >> "$detail_log"
            fi
        fi
        
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$timestamp,$total_cpu,$total_mem,$total_vmem,$total_threads,$process_count,$gpu_memory,$gpu_util,$ane_usage" >> "$resource_log"
        
        sleep 5
    done
}

# Helper function to check if process is running
is_process_running() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# For Apple Silicon, start powermetrics in background before monitoring
start_powermetrics() {
    local pid=$1
    if [ "$(uname -p)" = "arm" ]; then
         sudo powermetrics \
                -i 1000 \
                --samplers cpu_power,gpu_power,ane_power \
                --show-process-gpu \
                --format text \
                 > "/tmp/powermetrics_$pid.txt" 2>/dev/null &
        echo $!
    fi
}

# Stop powermetrics
stop_powermetrics() {
    local powermetrics_pid=$1
    if [ ! -z "$powermetrics_pid" ]; then
        echo sudo  kill $powermetrics_pid 2>/dev/null
    fi
}


# Function to run single experiment
run_experiment() {
    local config_file=$1
    local config_name=$(basename "$config_file" .yaml)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="$LOG_DIR/${config_name}_${timestamp}.log"
    local resource_log="$RESOURCE_DIR/${config_name}_${timestamp}_resources.csv"

    log_message "Starting experiment: $config_file at $(date)"
    log_message "Log file: $log_file"
    log_message "Resource monitoring: $resource_log"

    # Activate conda environment
    if [[ -f "$(conda info --base)/etc/profile.d/conda.sh" ]]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate $PYTHON_ENV || {
            log_message "Error: Failed to activate conda environment"
            return 1
        }
    else
        log_message "Error: Conda not found"
        return 1
    fi

    # Start the Python process
    python src/main.py --config $config_file --log DEBUG > $log_file 2>&1 &
    local python_pid=$!
    log_message "Started Python process with PID: $python_pid"

    # Initialize GPU monitoring if available
    local powermetrics_pid=""
    if [[ "$(uname -p)" == "arm" ]]; then
        powermetrics_pid=$(start_powermetrics $python_pid)
        if [[ -n "$powermetrics_pid" ]]; then
            log_message "Started powermetrics monitoring with PID: $powermetrics_pid"
        fi
    fi

    # Start resource monitoring
    monitor_resources $python_pid $resource_log &
    local monitor_pid=$!
    log_message "Started resource monitoring with PID: $monitor_pid"

    # Setup timeout
    timeout_duration=72000  # 20 hours in seconds
    end_time=$((SECONDS + timeout_duration))
    status=0

    log_message "Waiting for experiment completion (timeout: ${timeout_duration}s)..."
    
    # Wait for Python process with timeout
    while [ $SECONDS -lt $end_time ]; do
        if ! kill -0 $python_pid 2>/dev/null; then
            # Process has finished
            wait $python_pid
            status=$?
            log_message "Python process completed with status: $status"
            break
        fi
        
        # Check if the process is actually doing something
        if [ -f "$log_file" ]; then
            log_size=$(stat -f %z "$log_file")
            log_message "Current log size: $log_size bytes"
        fi
        
        sleep 30  # Check every 30 seconds
    done

    # Check if we timed out
    if [ $SECONDS -ge $end_time ]; then
        log_message "Experiment timed out after ${timeout_duration} seconds"
        status=124
    fi
    # Clean up processes
    log_message "Cleaning up processes..."
    
    # Stop monitoring processes
    if [[ -n "$monitor_pid" ]] && kill -0 $monitor_pid 2>/dev/null; then
        log_message "Stopping monitor process: $monitor_pid"
        kill -TERM $monitor_pid 2>/dev/null || true
        wait $monitor_pid 2>/dev/null || true
    fi

    # Stop powermetrics if running
    if [[ -n "$powermetrics_pid" ]]; then
        log_message "Stopping powermetrics process: $powermetrics_pid"
        if sudo -n true 2>/dev/null; then
            sudo -n kill -TERM $powermetrics_pid 2>/dev/null || true
        else
            log_message "Warning: Cannot stop powermetrics without sudo"
        fi
    fi

    # Kill Python process and its children if still running
    if kill -0 $python_pid 2>/dev/null; then
        log_message "Terminating Python process and children"
        pkill -P $python_pid 2>/dev/null || true
        kill -TERM $python_pid 2>/dev/null || true
        sleep 1
        kill -KILL $python_pid 2>/dev/null || true
    fi

    # Check results
    if [[ $status -eq 0 ]]; then
        if [[ -f "$log_file" && -s "$log_file" ]]; then
            log_message "Experiment completed successfully"
            log_message "$config_file:COMPLETED:$(date '+%Y-%m-%d %H:%M:%S')" >> $PROGRESS_FILE
            return 0
        else
            log_message "Error: Log file is empty or missing"
            return 1
        fi
    else
        log_message "Experiment failed with status: $status"
        local error_msg=$(tail -n 5 $log_file | tr '\n' ' ')
        log_message "$config_file:FAILED:$error_msg:$(date '+%Y-%m-%d %H:%M:%S')" >> $FAILED_FILE
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
