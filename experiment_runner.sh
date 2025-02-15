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
        sudo powermetrics -i 1000 --show-gpu --show-ane > "/tmp/powermetrics_$pid.txt" 2>/dev/null &
        echo $!
    fi
}

# Stop powermetrics
stop_powermetrics() {
    local powermetrics_pid=$1
    if [ ! -z "$powermetrics_pid" ]; then
        sudo kill $powermetrics_pid 2>/dev/null
    fi
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

    # Initialize GPU monitoring if available
    local powermetrics_pid=""
    if [ "$(uname -p)" = "arm" ]; then
        powermetrics_pid=$(start_powermetrics $python_pid)
        echo "Started powermetrics monitoring with PID: $powermetrics_pid"
    fi

    # Start resource monitoring
    monitor_resources $python_pid "$resource_log" &
    local monitor_pid=$!
    echo "Started resource monitoring with PID: $monitor_pid"

    # Function to cleanup processes
    cleanup_processes() {
        local pids_to_kill=("$@")
        for pid in "${pids_to_kill[@]}"; do
            if [ ! -z "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Killing process $pid"
                kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
            fi
        done
    }

    # Setup timeout
    local timeout_duration=7200  # 2 hours in seconds
    local end_time=$((SECONDS + timeout_duration))

    # Wait for Python process with timeout
    local status=0
    while [ $SECONDS -lt $end_time ]; do
        if ! kill -0 $python_pid 2>/dev/null; then
            wait $python_pid
            status=$?
            break
        fi
        sleep 5
    done

    # Check if we timed out
    if [ $SECONDS -ge $end_time ]; then
        echo "Experiment timed out after ${timeout_duration} seconds"
        status=124  # Traditional timeout exit code
    fi

    # Stop monitoring processes
    cleanup_processes $monitor_pid $powermetrics_pid

    # Stop powermetrics if running
    if [ ! -z "$powermetrics_pid" ]; then
        stop_powermetrics $powermetrics_pid
        rm -f "/tmp/powermetrics_$python_pid.txt"
    fi

    # Kill any remaining child processes
    pkill -P $python_pid 2>/dev/null
    cleanup_processes $python_pid

    # Generate resource usage plots
    if [ -f "$resource_log" ] && [ -s "$resource_log" ]; then
        generate_resource_plots "$resource_log"
    fi

    # Process results
    if [ $status -eq 0 ]; then
        echo "$config_file:COMPLETED:$(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
        echo "Successfully completed: $config_file"
        
        # Generate summary statistics
        {
            echo "Experiment Summary"
            echo "=================="
            echo "Config: $config_name"
            echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Duration: $SECONDS seconds"
            echo ""
            echo "Resource Usage Summary"
            echo "---------------------"
            awk -F',' '
                NR>1 {
                    cpu+=$2; mem+=$3; vmem+=$4; threads+=$5; count++
                    if($2>max_cpu) max_cpu=$2
                    if($3>max_mem) max_mem=$3
                }
                END {
                    if(count>0) {
                        printf "Average CPU: %.2f%%\n", cpu/count
                        printf "Average Memory: %.2f MB\n", mem/count
                        printf "Peak CPU: %.2f%%\n", max_cpu
                        printf "Peak Memory: %.2f MB\n", max_mem
                    }
                }
            ' "$resource_log" 
        } > "${resource_log%.*}_summary.txt"
        
        return 0
    else
        local error_msg=$(tail -n 5 "$log_file" | tr '\n' ' ')
        echo "$config_file:FAILED:$error_msg:$(date '+%Y-%m-%d %H:%M:%S')" >> "$FAILED_FILE"
        echo "Failed: $config_file"
        echo "Exit status: $status"
        echo "Last few lines of log:"
        tail -n 10 "$log_file"
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
