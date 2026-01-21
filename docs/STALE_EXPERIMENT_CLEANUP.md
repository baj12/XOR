# Automatic Stale Experiment Cleanup

**Date**: 2026-01-05
**Status**: ✅ Complete and Active

## Overview

Automatic detection and cleanup of experiments that are marked as "running" but have no actual orchestrator process. This handles scenarios where the system restarts, hibernates, or crashes while experiments are in progress.

## Problem Statement

**Scenario**: Computer restarts or goes into hibernation while a continuous experiment is running.

**Before**:
- Experiment remains in database with `status='running'`
- No actual orchestrator process exists (PID invalid/null)
- User sees "running" in UI but nothing is happening
- Manual intervention required to fix database state

**After**:
- Automatic detection on Flask startup
- Periodic checking every 5 minutes
- Database automatically updated to `status='stopped'`
- Alert logged for audit trail
- No manual intervention needed

## How It Works

### Detection Logic

```python
def is_stale_experiment(experiment):
    """An experiment is stale if marked 'running' but process doesn't exist"""
    if experiment.status != 'running':
        return False

    if experiment.orchestrator_pid is None:
        return True  # No PID stored

    try:
        os.kill(pid, 0)  # Check if process exists
        return False  # Process exists - not stale
    except (OSError, ProcessLookupError):
        return True  # Process doesn't exist - stale!
```

### Cleanup Actions

When a stale experiment is detected:

1. **Update Database Status**:
   ```sql
   UPDATE continuous_experiments
   SET status = 'stopped',
       orchestrator_pid = NULL,
       updated_at = NOW()
   WHERE experiment_id = ?
   ```

2. **Log Alert**:
   ```sql
   INSERT INTO experiment_alerts
   (experiment_id, alert_type, severity, message, details)
   VALUES (?, 'other', 'warning', 'Stale process cleanup...', ?)
   ```

3. **Log to Console**:
   ```
   ⚠ Stale experiment detected: exp_xxx (Process 12345 not running)
   → Cleaned up exp_xxx: status='stopped', PID cleared
   ```

## Implementation Components

### 1. Standalone Script

**File**: [scripts/cleanup_stale_experiments.py](../scripts/cleanup_stale_experiments.py)

**Usage**:
```bash
# Dry run (shows what would be done)
python scripts/cleanup_stale_experiments.py --dry-run

# Actually clean up
python scripts/cleanup_stale_experiments.py
```

**Features**:
- Standalone operation (doesn't require Flask)
- Can be run manually or via cron
- Comprehensive logging and statistics
- Dry-run mode for safety

**Example Output**:
```
============================================================
Stale Experiment Cleanup
============================================================
Found 1 experiments marked as 'running'
⚠ exp_4310cb8d: STALE - No PID stored
  → Cleaned up: set status='stopped', cleared PID
============================================================
SUMMARY
============================================================
Experiments checked: 1
Stale experiments found: 1
Cleaned up: 1 experiments
```

### 2. Flask Integration

**File**: [web/app.py](../web/app.py)

**Automatic Cleanup Function** (lines 50-138):
```python
def cleanup_stale_experiments_on_startup():
    """
    Cleanup experiments marked as 'running' but with no actual process.

    This handles cases where the system restarted/hibernated while experiments
    were running, leaving stale 'running' status in the database.
    """
```

**Called On Startup** (line 1476):
```python
# Run stale experiment cleanup on startup (handles system restarts/hibernation)
cleanup_stale_experiments_on_startup()
```

**Periodic Checking** (lines 142-156):
```python
# Setup periodic cleanup scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_stale_experiments_on_startup,
    trigger='interval',
    minutes=5,
    id='stale_experiment_cleanup',
    name='Periodic stale experiment cleanup',
    replace_existing=True
)
scheduler.start()
```

## When Cleanup Runs

### Startup (Always)
- Every time Flask starts
- Cleans up stale experiments from previous session
- Ensures clean state before serving requests

### Periodic (Every 5 Minutes)
- Runs in background via APScheduler
- Catches experiments that become stale during operation
- Handles edge cases (process crashes, OOM kills, etc.)

### Manual (On Demand)
- Run standalone script anytime
- Useful for debugging or verification
- Can use `--dry-run` to preview without changes

## Dependencies

**Python Packages**:
- `apscheduler` - Background task scheduling
- `mysql-connector-python` - Database access
- Standard library: `os`, `signal`, `logging`, `json`

**Installation**:
```bash
pip install apscheduler
```

## Logs and Alerts

### Console Logs

**Normal Operation** (no stale experiments):
```
INFO - Running stale experiment cleanup...
INFO - No running experiments found - nothing to clean up
```

**Stale Experiment Found**:
```
INFO - Running stale experiment cleanup...
WARNING - ⚠ Stale experiment detected: exp_xxx (Process 12345 not running)
INFO -   → Cleaned up exp_xxx: status='stopped', PID cleared
INFO - ✓ Cleanup complete: 1 stale experiment(s) cleaned
```

**Valid Running Experiment**:
```
INFO - Running stale experiment cleanup...
INFO - ✓ exp_yyy: Process 54321 is running
INFO - ✓ All running experiments have valid processes
```

### Database Alerts

Every cleanup creates an alert in `experiment_alerts` table:

```sql
SELECT * FROM experiment_alerts
WHERE alert_type = 'other'
AND message LIKE 'Stale process cleanup%'
ORDER BY created_at DESC;
```

**Example Alert**:
- `experiment_id`: exp_4310cb8d
- `alert_type`: other
- `severity`: warning
- `message`: Stale process cleanup: Experiment marked as stopped due to missing process. No PID stored
- `details`: JSON with experiment_id, pid, reason, cleanup_time

## Scenarios Handled

### 1. System Restart
**What Happens**:
- Experiment running with PID 12345
- System restarts (power loss, forced reboot, etc.)
- PID 12345 no longer exists

**Cleanup Action**:
- Detects PID 12345 doesn't exist
- Updates status to 'stopped'
- Logs alert: "Process 12345 not running"

### 2. System Hibernation
**What Happens**:
- Experiment running with PID 23456
- System hibernates for hours/days
- Process killed by OS during hibernation

**Cleanup Action**:
- On wake/startup, detects PID 23456 dead
- Updates status to 'stopped'
- Logs alert with reason

### 3. Missing PID
**What Happens**:
- Experiment status='running' but orchestrator_pid=NULL
- Could happen if database update failed during start

**Cleanup Action**:
- Detects NULL PID for running experiment
- Updates status to 'stopped'
- Logs alert: "No PID stored"

### 4. Process Crash
**What Happens**:
- Experiment running normally
- Orchestrator crashes (OOM, segfault, etc.)
- Process dies but database still shows 'running'

**Cleanup Action**:
- Periodic check detects dead PID
- Updates status to 'stopped' within 5 minutes
- Logs alert for investigation

## Testing

### Test Stale Experiment Cleanup

**1. Create a stale experiment**:
```sql
-- Manually set an experiment to running with invalid PID
UPDATE continuous_experiments
SET status = 'running',
    orchestrator_pid = 99999
WHERE experiment_id = 'exp_test';
```

**2. Run cleanup script**:
```bash
python scripts/cleanup_stale_experiments.py
```

**Expected Output**:
```
⚠ exp_test: STALE - Process 99999 not running
→ Cleaned up: set status='stopped', cleared PID
```

**3. Verify database**:
```sql
SELECT experiment_id, status, orchestrator_pid
FROM continuous_experiments
WHERE experiment_id = 'exp_test';

-- Should show: status='stopped', orchestrator_pid=NULL
```

### Test Flask Integration

**1. Stop Flask if running**:
```bash
pkill -f "python.*web/app.py"
```

**2. Create stale experiment** (as above)

**3. Start Flask**:
```bash
python web/app.py
```

**4. Check logs** - should see cleanup happen immediately on startup

**5. Verify periodic execution** - wait 5 minutes, should run again

## Configuration

### Cleanup Interval

To change how often periodic cleanup runs, edit [web/app.py](../web/app.py) line 146:

```python
scheduler.add_job(
    func=cleanup_stale_experiments_on_startup,
    trigger='interval',
    minutes=5,  # Change this value
    ...
)
```

**Recommendations**:
- **Production**: 5 minutes (default) - good balance
- **Development**: 1 minute - faster detection during testing
- **Low Activity**: 15 minutes - reduce overhead if experiments rarely run

### Disable Periodic Cleanup

To disable periodic cleanup but keep startup cleanup:

Comment out the scheduler section in [web/app.py](../web/app.py):

```python
# # Setup periodic cleanup scheduler
# scheduler = BackgroundScheduler()
# scheduler.add_job(...)
# scheduler.start()
```

## Troubleshooting

### Cleanup Not Running

**Symptoms**: Stale experiments remain in database

**Check**:
1. Flask logs on startup - should see "Running stale experiment cleanup..."
2. Check scheduler: `scheduler.get_jobs()` should show the cleanup job
3. Verify APScheduler installed: `pip list | grep apscheduler`

**Fix**:
```bash
# Reinstall APScheduler
pip install --upgrade apscheduler

# Restart Flask
pkill -f "python.*web/app.py"
python web/app.py
```

### Too Many Alerts

**Symptoms**: Alert table filling up with cleanup messages

**Cause**: Experiments constantly becoming stale (shouldn't happen normally)

**Investigate**:
```sql
-- Check cleanup frequency
SELECT DATE(created_at) as date, COUNT(*) as cleanups
FROM experiment_alerts
WHERE message LIKE 'Stale process cleanup%'
GROUP BY DATE(created_at)
ORDER BY date DESC;
```

**Root Cause Investigation**:
- Are processes crashing frequently? (check orchestrator logs)
- Is system hibernating/restarting often? (check system logs)
- Are experiments being killed externally? (check for OOM killer)

### Cleanup Too Aggressive

**Symptoms**: Valid experiments marked as stopped

**Cause**: Should never happen - `os.kill(pid, 0)` is reliable

**Debug**:
```bash
# Run dry-run to see what would be cleaned
python scripts/cleanup_stale_experiments.py --dry-run

# Check specific PID
ps -p <pid>  # Should show nothing if process dead
```

## Benefits

✅ **Zero Manual Intervention**: System self-heals after restarts/hibernation
✅ **Audit Trail**: Every cleanup logged in experiment_alerts
✅ **Fast Detection**: 5-minute maximum delay to detect stale experiments
✅ **Safe Operation**: Uses `os.kill(pid, 0)` which never harms processes
✅ **Comprehensive Coverage**: Handles restarts, crashes, hibernation, missing PIDs
✅ **Standalone Tool**: Can run cleanup manually or via cron
✅ **Production Ready**: Integrated into Flask with proper error handling

## Files Modified

| File | Changes |
|------|---------|
| `scripts/cleanup_stale_experiments.py` | NEW - Standalone cleanup script |
| `web/app.py` | Added cleanup function, startup call, periodic scheduler |
| `docs/STALE_EXPERIMENT_CLEANUP.md` | NEW - This documentation |

## Related Documentation

- [Process Management](PROCESS_MANAGEMENT.md) - Overall process lifecycle documentation
- [Continuous Learning Status](CONTINUOUS_LEARNING_STATUS.md) - System status
- [MariaDB Schema](MARIADB_SCHEMA.md) - Database schema including experiment_alerts

## Future Enhancements

- [ ] Web UI dashboard showing cleanup statistics
- [ ] Email notifications for stale experiment detection
- [ ] Configurable cleanup interval via environment variable
- [ ] Cleanup metrics API endpoint
- [ ] Auto-restart option for experiments that became stale
