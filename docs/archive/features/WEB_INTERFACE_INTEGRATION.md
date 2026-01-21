# Web Interface Integration for Orchestrator

## Summary

The continuous learning orchestrator now automatically registers itself with the MariaDB database and appears on the web interface.

## Changes Made

### 1. Current Test Run Registration (Immediate Fix)

**File:** `register_test_run.py`

- Manually registered the currently running 24hr test in MariaDB
- Experiment ID: `test_24hr_20260121`
- Status: `running`
- PID: `69418`
- The test is NOW visible on the web interface

### 2. Orchestrator Auto-Registration (Future Runs)

**File:** `src/continuous/orchestrator.py`

**Added Features:**

#### a. Web Integration Parameter
- New `web_integration` parameter in `__init__()` (default: `True`)
- Automatically connects to MariaDB if enabled
- Falls back gracefully if MariaDB unavailable

#### b. Auto-Registration on Startup
- New method: `_register_with_web_interface()`
- Called automatically when `run()` starts
- Creates entry in `continuous_experiments` table with:
  - Unique experiment ID: `orchestrator_YYYYMMDD_HHMMSS`
  - Current PID for process tracking
  - Start/end times
  - Configuration parameters
  - Status: `running`

#### c. Real-Time Status Updates
- New method: `_update_web_interface()`
- Called every orchestration cycle (every 5 minutes)
- Updates in MariaDB:
  - Total samples collected
  - Current status
  - Updated timestamp
  - Any additional status info

#### d. Completion/Stop Handling
- Updates status to `completed` when test duration finishes
- Updates status to `stopped` when manually stopped
- Clears PID when stopped

## Usage

### For Future Test Runs

Simply start the orchestrator as usual - it will automatically register:

```bash
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/features.db \
    --model-dir models/continuous \
    --report-dir reports/continuous \
    --test-hours 24
```

The experiment will immediately appear on the web interface!

### Disable Web Integration (Optional)

If you want to run without web interface integration:

```python
orchestrator = ContinuousLearningOrchestrator(
    config=config,
    db_path=db_path,
    model_dir=model_dir,
    report_dir=report_dir,
    web_integration=False  # Disable MariaDB integration
)
```

## Web Interface Display

The web interface (http://localhost:5001) now shows:

- **Experiment ID**: `test_24hr_20260121` or `orchestrator_YYYYMMDD_HHMMSS`
- **Name**: Descriptive name with timestamp
- **Status**: `running`, `completed`, or `stopped`
- **PID**: Current process ID (allows checking if still alive)
- **Samples Collected**: Updated every 5 minutes
- **Start/End Times**: Full timeline visibility
- **Configuration**: Training parameters, provider type, etc.

## Database Schema

Updates are stored in the `continuous_experiments` table in MariaDB:

```sql
SELECT experiment_id, experiment_name, status, orchestrator_pid,
       start_time, total_samples_collected
FROM continuous_experiments
WHERE status = 'running'
ORDER BY start_time DESC;
```

## Benefits

1. **Visibility**: All orchestrator runs visible on web interface
2. **Monitoring**: Real-time sample counts and status updates
3. **Process Tracking**: PID tracking enables process health checks
4. **History**: Complete audit trail of all runs
5. **Integration**: Seamless with existing web UI for recording experiments

## Current Test Status

✅ **The running 24hr test (PID 69418) is NOW visible on the web interface!**

Check it at: http://localhost:5001

Look for experiment ID: `test_24hr_20260121`

## Backward Compatibility

- Web integration is enabled by default but fails gracefully if MariaDB unavailable
- SQLite-only mode still works (just won't show on web interface)
- Existing code/tests not affected
- No breaking changes to orchestrator API

## Testing

To verify integration:

```bash
# Check MariaDB for your experiment
source /Users/bernd/miniconda3/bin/activate xorProject
python3 -c "
from src.db_connection import DatabaseConnection
db = DatabaseConnection(backend='mariadb')
with db.get_connection() as conn:
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM continuous_experiments WHERE status=\"running\"')
    print(cursor.fetchall())
"
```

## Next Steps

Future enhancements could include:
- More detailed progress metrics (cycles completed, files processed, etc.)
- Performance graphs over time
- Alert integration (link alerts to experiments)
- Download logs/results through web interface
- Pause/resume controls from web UI
