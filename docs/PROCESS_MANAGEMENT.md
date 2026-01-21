# Process Management for Continuous Experiments

**Date**: 2026-01-04
**Status**: ✅ Complete

## Overview

Implemented comprehensive process management for continuous learning experiments, including PID tracking, graceful pause, and force kill capabilities.

## Problem Statement

**Before**:
- Pause set database flag but didn't verify orchestrator stopped
- No way to force-kill stuck experiments
- PIDs not tracked - couldn't manage background processes
- No visibility into whether orchestrator actually responded to pause

**After**:
- PIDs stored in database for tracking
- Pause verifies process status and orchestrator checks database
- Kill endpoint forcefully terminates processes
- Full process lifecycle management

## Changes Made

### 1. Database Schema

Added PID tracking column:

```sql
ALTER TABLE continuous_experiments
ADD COLUMN orchestrator_pid INT NULL AFTER status;
```

**Purpose**: Track which process is running each experiment for management.

### 2. API Endpoints

#### Updated: `POST /api/continuous/experiments/<id>/start`

Now stores PID in database when starting:

```python
# Store PID in database
cursor.execute("""
    UPDATE continuous_experiments
    SET orchestrator_pid = %s,
        status = 'running',
        updated_at = NOW()
    WHERE experiment_id = %s
""", (process.pid, experiment_id))
```

**Response**:
```json
{
  "success": true,
  "message": "Experiment started",
  "pid": 12345,
  "log_file": "/path/to/exp_xxx.log"
}
```

#### Enhanced: `POST /api/continuous/experiments/<id>/pause`

Now checks if process is actually running:

```python
# Check if process exists
try:
    os.kill(pid, 0)  # Signal 0 just checks existence
    process_running = True
except (OSError, ProcessLookupError):
    process_running = False
```

**Response**:
```json
{
  "success": true,
  "message": "Experiment pause requested (will stop after current cycle)",
  "pid": 12345,
  "process_still_running": true
}
```

**Behavior**:
- Sets database status to `'paused'`
- Orchestrator detects this and stops gracefully
- Returns PID and whether process is still running

#### New: `POST /api/continuous/experiments/<id>/kill`

Force terminates the orchestrator process:

```python
# Send SIGTERM to process
os.kill(pid, signal.SIGTERM)

# Update database
cursor.execute("""
    UPDATE continuous_experiments
    SET status = 'stopped',
        orchestrator_pid = NULL,
        updated_at = NOW()
    WHERE experiment_id = %s
""", (experiment_id,))
```

**Response**:
```json
{
  "success": true,
  "message": "Experiment killed",
  "pid": 12345,
  "killed": true,
  "previous_status": "running",
  "warning": null
}
```

**Error Handling**:
- If process already dead: `killed: false, warning: "Process not found"`
- If permission denied: Returns 403 error
- Always updates database status regardless

### 3. Orchestrator Changes

#### Added: Pause Detection

Orchestrator now checks database status every cycle:

```python
def _check_pause_request(self):
    """Check database for pause request from user"""
    with self.db.get_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT status FROM continuous_experiments
            WHERE experiment_id = %s
        """, (self.experiment_id,))

        result = cursor.fetchone()
        if result and result['status'] == 'paused':
            self.logger.info("Pause detected from database")
            self.should_stop = True
```

**Called at start of each cycle**:
```python
while self.current_cycle < total_cycles:
    # Check for pause before starting new cycle
    self._check_pause_request()

    if self.should_stop:
        self.update_experiment_status('paused', orchestrator_pid=None)
        break
```

#### PID Cleanup on Exit

Orchestrator clears its PID when stopping:

```python
# On pause
self.update_experiment_status('paused', orchestrator_pid=None)

# On completion
self.update_experiment_status('completed',
                               completed_at=datetime.now(),
                               orchestrator_pid=None)
```

### 4. Web Interface

Added Force Kill button to experiment modal:

```html
<button type="button" class="btn btn-danger" id="killBtn" style="display:none;">
    <i class="bi bi-x-octagon"></i> Force Kill
</button>
```

**Button Visibility**:
- **Running**: Pause + Kill buttons shown
- **Paused/Stopped**: Start button shown
- **Completed/Failed**: No action buttons

**JavaScript**:
```javascript
async function killExperiment(experimentId) {
    if (!confirm('⚠️ WARNING: Force kill will immediately terminate!\n\n...')) return;

    const response = await fetch(`/api/continuous/experiments/${experimentId}/kill`, {
        method: 'POST'
    });
    const data = await response.json();

    if (data.success) {
        const msg = data.killed ?
            `Experiment killed (PID ${data.pid} terminated).` :
            `Experiment stopped. ${data.warning}`;
        alert(msg);
    }
}
```

## Usage

### Check Running Experiments

Query database to see which experiments have active processes:

```sql
SELECT experiment_id, status, orchestrator_pid
FROM continuous_experiments
WHERE orchestrator_pid IS NOT NULL;
```

### Pause an Experiment (Graceful)

**Via API**:
```bash
curl -X POST http://localhost:5001/api/continuous/experiments/exp_xxx/pause
```

**Via UI**:
1. Open experiment details
2. Click "Pause" button
3. Wait for current cycle to complete

**Result**:
- Orchestrator sees `status = 'paused'` in database
- Completes current cycle
- Exits gracefully
- Clears PID from database

### Kill an Experiment (Force)

**Via API**:
```bash
curl -X POST http://localhost:5001/api/continuous/experiments/exp_xxx/kill
```

**Via UI**:
1. Open experiment details
2. Click "Force Kill" button (red)
3. Confirm warning dialog

**Result**:
- SIGTERM sent to process immediately
- Database updated to `'stopped'`
- PID cleared
- May interrupt current cycle

### Verify Process Stopped

```bash
# Check if PID exists
ps -p <pid>

# Or check database
curl http://localhost:5001/api/continuous/experiments/exp_xxx | grep orchestrator_pid
```

## Process Lifecycle

```
┌─────────────┐
│   Created   │ (status: stopped, pid: NULL)
│             │
└──────┬──────┘
       │
       ▼ POST /start
┌─────────────┐
│   Running   │ (status: running, pid: 12345)
│             │
└──────┬──────┘
       │
       ├─── POST /pause ───────────┐
       │                           │
       ▼                           ▼
┌─────────────┐            ┌─────────────┐
│   Paused    │            │   Stopped   │
│ (graceful)  │            │ (immediate) │
│ pid: NULL   │◄── /kill ─┤  pid: NULL  │
└─────────────┘            └─────────────┘
       │
       ▼ POST /start (resume)
┌─────────────┐
│   Running   │
└─────────────┘
```

## Troubleshooting

### Pause Not Working

**Symptoms**: Experiment status shows "paused" but process still running

**Check**:
```bash
# Get PID from database
curl http://localhost:5001/api/continuous/experiments/exp_xxx | grep orchestrator_pid

# Check if process exists
ps -p <pid>
```

**Solution**: Use Force Kill

### Kill Returns "Process Not Found"

**Cause**: Process already stopped but database still had stale PID

**Behavior**: API will still update database status to `'stopped'` and clear PID

**No action needed**: This is normal cleanup

### Permission Denied on Kill

**Cause**: Process owned by different user

**Solution**:
- Check process owner: `ps -o user= -p <pid>`
- Run Flask as same user
- Or manually kill: `sudo kill <pid>`

## Files Modified

| File | Changes |
|------|---------|
| `continuous_experiments` table | Added `orchestrator_pid` column |
| `web/app.py` | - Store PID on start<br>- Enhanced pause with status check<br>- New kill endpoint |
| `src/continuous/recording_orchestrator.py` | - Check database for pause<br>- Clear PID on exit |
| `web/templates/continuous_dashboard.html` | - Add Kill button<br>- JavaScript kill function |

## API Reference

### Start Experiment
- **Endpoint**: `POST /api/continuous/experiments/<id>/start`
- **Response**: `{success, message, pid, log_file}`
- **Side Effect**: Sets `status='running'`, `orchestrator_pid=<pid>`

### Pause Experiment
- **Endpoint**: `POST /api/continuous/experiments/<id>/pause`
- **Response**: `{success, message, pid, process_still_running}`
- **Side Effect**: Sets `status='paused'`, orchestrator detects and stops

### Kill Experiment
- **Endpoint**: `POST /api/continuous/experiments/<id>/kill`
- **Response**: `{success, message, pid, killed, previous_status, warning}`
- **Side Effect**: SIGTERM to process, sets `status='stopped'`, `orchestrator_pid=NULL`

## Benefits

✅ **Full Process Control**: Can start, pause, and kill experiments
✅ **PID Tracking**: Know which processes are running
✅ **Graceful Shutdown**: Pause completes current cycle before stopping
✅ **Force Termination**: Kill works when pause fails
✅ **Process Verification**: APIs report if process actually running
✅ **Clean State**: PIDs cleared when experiments stop
✅ **User-Friendly**: Clear UI with warnings for destructive actions
