# Rubix44 Server Update Instructions

**Server**: 10.0.0.58:5000
**Repository**: https://github.com/baj12/rubix44-recorder.git
**Date**: 2026-01-09

## Current Status

**Rubix44 server is RUNNING but has a stale recording session:**
- Session ID: `20260105_094836`
- Started: 2026-01-05 09:48:36
- Duration: 93+ hours (running since experiment that crashed)
- Status: Recording (but probably not actually recording)

## Update Procedure

### Step 1: SSH to Rubix44 Server

```bash
# From your Mac, SSH to the rubix44 server
ssh bernd@10.0.0.58

# Or if using a different username:
ssh <username>@10.0.0.58
```

### Step 2: Stop the Stale Recording Session

```bash
# Stop the current recording via API
curl -X POST http://localhost:5000/api/v1/recordings/stop

# Verify it stopped
curl http://localhost:5000/api/v1/recordings/status
# Should show: "status": "idle"
```

### Step 3: Locate the Rubix44 Installation

```bash
# Find where rubix44-recorder is installed
# Common locations:
cd ~/rubix44-recorder
# OR
cd /opt/rubix44-recorder
# OR
cd /home/bernd/rubix44-recorder

# Verify you're in the right directory
ls -la
# Should see: api_server.py, rubix_recorder.py, environment.yml, etc.
```

### Step 4: Stop the API Server

**If running as systemd service:**
```bash
sudo systemctl stop rubix-recorder-api
```

**If running in screen/tmux:**
```bash
# List screen sessions
screen -ls
# Attach to the session
screen -r <session_name>
# Press Ctrl+C to stop the server
# Detach: Ctrl+A then D
```

**If running in background:**
```bash
# Find the process
ps aux | grep api_server.py

# Kill it
pkill -f api_server.py
```

### Step 5: Backup Current Installation

```bash
# Create backup of current version
cd ~/
tar -czf rubix44-recorder-backup-$(date +%Y%m%d).tar.gz rubix44-recorder/

# Verify backup created
ls -lh rubix44-recorder-backup-*.tar.gz
```

### Step 6: Update the Code

```bash
cd rubix44-recorder

# Check current branch/version
git branch
git log -1 --oneline

# Fetch latest changes
git fetch origin

# Pull latest code
git pull origin main

# Check what changed
git log --oneline -10
```

### Step 7: Update Dependencies

**If using conda (recommended):**
```bash
# Activate the environment
conda activate rubix-recorder-api

# Update environment
conda env update -f environment.yml --prune

# Verify packages updated
conda list | grep -E "flask|sounddevice|numpy"
```

**If using pip:**
```bash
pip install --upgrade -r requirements.txt
```

### Step 8: Restart the Server

**If using systemd:**
```bash
sudo systemctl start rubix-recorder-api
sudo systemctl status rubix-recorder-api
```

**If using screen:**
```bash
screen -S rubix-api
conda activate rubix-recorder-api
python api_server.py
# Detach: Ctrl+A then D
```

**If using startup script:**
```bash
# On Linux/Mac
./start_api_server.sh

# On Windows
start_api_server.bat
```

### Step 9: Verify Update

```bash
# From the rubix44 server or your Mac:

# Check health
curl http://10.0.0.58:5000/api/v1/health

# Check status (should be idle)
curl http://10.0.0.58:5000/api/v1/recordings/status

# Check config
curl http://10.0.0.58:5000/api/v1/config

# List available playback files
curl http://10.0.0.58:5000/api/v1/playback-files
```

### Step 10: Test Recording

```bash
# Start a short test recording
curl -X POST http://10.0.0.58:5000/api/v1/recordings/start \
  -H "Content-Type: application/json" \
  -d '{
    "playback_file": "elvisBlackA.wav",
    "duration": 60,
    "output_prefix": "test"
  }'

# Wait 60+ seconds, then check status
curl http://10.0.0.58:5000/api/v1/recordings/status

# Should show: "status": "idle" after completion

# Check recording history
curl http://10.0.0.58:5000/api/v1/recordings/history
```

## Quick Update (No Code Changes Needed)

If you just want to restart the server without updating code:

```bash
ssh bernd@10.0.0.58

# Stop stale recording
curl -X POST http://localhost:5000/api/v1/recordings/stop

# Restart API server
pkill -f api_server.py
sleep 2
cd rubix44-recorder
conda activate rubix-recorder-api
nohup python api_server.py > logs/api_server.log 2>&1 &

# Verify
curl http://localhost:5000/api/v1/recordings/status
```

## Configuration Files

### API Configuration
**Location**: `config/api_config.json`

```json
{
  "host": "0.0.0.0",
  "port": 5000,
  "debug": false,
  "default_duration": 3600,
  "sample_rate": 44100,
  "output_prefix": "api_recording",
  "playback_directory": "playback_files",
  "recordings_directory": "recordings"
}
```

**Important Settings:**
- `host: "0.0.0.0"` - Listens on all interfaces (needed for network access)
- `port: 5000` - Must match XOR configuration
- `sample_rate: 44100` - Must match Rubix44 hardware

## Troubleshooting

### Server Won't Start

**Check logs:**
```bash
tail -100 logs/api_server.log
```

**Common issues:**
- Port 5000 already in use: `lsof -i :5000` or change port in config
- Conda environment not activated: `conda activate rubix-recorder-api`
- Missing dependencies: `conda env update -f environment.yml`

### Rubix44 Device Not Found

```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Or via API
curl http://localhost:5000/api/v1/devices

# Check USB connection
lsusb | grep -i rubix  # Linux
system_profiler SPUSBDataType | grep -i rubix  # Mac
```

### Recordings Not Working

1. Check playback files exist:
   ```bash
   ls -la playback_files/
   ```

2. Check recordings directory permissions:
   ```bash
   ls -ld recordings/
   # Should be writable by the user running api_server.py
   ```

3. Test recording manually:
   ```bash
   python rubix_recorder.py --playback playback_files/elvisBlackA.wav --duration 10
   ```

### Network Issues

**From XOR system (10.11.0.126), test connectivity:**
```bash
# Ping test
ping -c 3 10.0.0.58

# Port test
nc -zv 10.0.0.58 5000

# HTTP test
curl http://10.0.0.58:5000/api/v1/health
```

**Firewall rules (if needed on rubix44 server):**
```bash
# Linux (ufw)
sudo ufw allow 5000/tcp

# Linux (firewalld)
sudo firewall-cmd --add-port=5000/tcp --permanent
sudo firewall-cmd --reload
```

## Checking for Updates

**See what's new in the repository:**
```bash
cd rubix44-recorder

# Fetch latest
git fetch origin

# Compare your version to latest
git log HEAD..origin/main --oneline

# See detailed changes
git log HEAD..origin/main

# See file changes
git diff HEAD..origin/main
```

## Rollback Procedure

If update causes issues:

```bash
cd rubix44-recorder

# Stop server
pkill -f api_server.py

# Restore from backup
cd ~/
tar -xzf rubix44-recorder-backup-YYYYMMDD.tar.gz

# Or git rollback
cd rubix44-recorder
git log --oneline -20  # Find commit to rollback to
git reset --hard <commit-hash>

# Restart server
conda activate rubix-recorder-api
nohup python api_server.py > logs/api_server.log 2>&1 &
```

## Maintenance Schedule

**Weekly:**
- Check disk space in `recordings/` directory
- Review logs for errors
- Verify server is responding to health checks

**Monthly:**
- Update dependencies: `conda env update -f environment.yml`
- Clean old recordings (if not needed)
- Backup configuration and important recordings

**When XOR System Issues Occur:**
1. Check rubix44 server is running: `curl http://10.0.0.58:5000/api/v1/health`
2. Stop any stale recordings: `curl -X POST http://10.0.0.58:5000/api/v1/recordings/stop`
3. Check logs: `tail -100 logs/api_server.log`
4. Restart if needed

## Integration with XOR System

The XOR continuous learning system connects via [src/continuous/rubix44_data_provider.py](../src/continuous/rubix44_data_provider.py).

**Default configuration:**
- Rubix44 URL: `http://10.0.0.58:5000`
- API base: `/api/v1`
- Timeout: 5 seconds

**Key endpoints used by XOR:**
- `GET /api/v1/recordings/status` - Check if recording in progress
- `POST /api/v1/recordings/start` - Start new recording
- `POST /api/v1/recordings/stop` - Stop current recording
- `GET /api/v1/recordings/history` - List completed recordings

## Contact and Support

**Repository**: https://github.com/baj12/rubix44-recorder
**Issues**: https://github.com/baj12/rubix44-recorder/issues
**Documentation**: See `API_DOCS.md` in repository

## Quick Reference Commands

```bash
# Status check
curl http://10.0.0.58:5000/api/v1/recordings/status

# Stop recording
curl -X POST http://10.0.0.58:5000/api/v1/recordings/stop

# Start recording
curl -X POST http://10.0.0.58:5000/api/v1/recordings/start \
  -H "Content-Type: application/json" \
  -d '{"playback_file": "elvisBlackA.wav", "duration": 3600, "output_prefix": "test"}'

# Health check
curl http://10.0.0.58:5000/api/v1/health

# List devices
curl http://10.0.0.58:5000/api/v1/devices

# View logs (on rubix44 server)
tail -f ~/rubix44-recorder/logs/api_server.log
```
