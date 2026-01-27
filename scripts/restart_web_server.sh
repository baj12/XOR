#!/bin/bash
# Restart the web server for the Recording Management System
# Usage: ./scripts/restart_web_server.sh [--debug]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=${PORT:-5001}

echo "=== Restarting Web Server ==="
echo "Project directory: $PROJECT_DIR"
echo "Port: $PORT"

# Change to project directory
cd "$PROJECT_DIR"

# Kill any existing web server processes
echo "Stopping existing server processes..."
pkill -f "python.*web/app.py" 2>/dev/null || true
pkill -f "python -m web.app" 2>/dev/null || true

# Wait for processes to die
sleep 1

# Check if port is still in use
if lsof -i :$PORT 2>/dev/null | grep -q LISTEN; then
    echo "Warning: Port $PORT is still in use. Attempting to free it..."
    lsof -i :$PORT | grep LISTEN | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Activate conda environment and start server
echo "Starting web server..."
source /Users/bernd/miniconda3/bin/activate xorProject

# Check for debug flag
if [[ "$1" == "--debug" ]]; then
    echo "Running in DEBUG mode..."
    python web/app.py --port $PORT --debug
else
    echo "Running in production mode..."
    python web/app.py --port $PORT &
    SERVER_PID=$!
    echo "Server started with PID: $SERVER_PID"
    sleep 2

    # Check if server is running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Server is running at http://localhost:$PORT"
        echo "To stop: kill $SERVER_PID"
    else
        echo "ERROR: Server failed to start. Check logs for details."
        exit 1
    fi
fi
