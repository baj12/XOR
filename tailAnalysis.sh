#!/bin/bash
# filepath: /Users/bernd/python/XOR/tailAnalysis.sh
# This script extracts the last two lines per PID from sorted_mylog.txt

# Check if 'tac' is available; if not, install it
if ! command -v tac &> /dev/null
then
    echo "'tac' could not be found. Installing 'coreutils' via Homebrew..."
    brew install coreutils
    # 'tac' is provided as 'gtac' by coreutils on macOS
    alias tac='gtac'
fi

# Process the log file
tac sorted_mylog.txt | awk '{
    pid = $4
    gsub(/\]$/, "", pid)
    if (pid != current_pid) {
        current_pid = pid
        count = 0
    }
    if (count < 2) {
        print $0
        count++
    }
}' | tac > last_two_lines_per_pid.txt

echo "Extraction complete. Check 'last_two_lines_per_pid.txt' for results."
