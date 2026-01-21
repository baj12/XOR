# Fractional Weeks Support - Fix Documentation

## Problem Summary

Experiment `exp_f11692c0` was configured with `target_duration_weeks = 0.01` but completed immediately without running any cycles.

### Root Cause

The database column `continuous_experiments.target_duration_weeks` was defined as `INT(11)`, which **truncated** decimal values:
- Input: `0.01` weeks
- Stored as: `0` (integer truncation)
- Result: 0 cycles calculated, experiment completed instantly

## Solution Applied

### 1. Database Migration ✓

Changed column type from `INT` to `FLOAT`:

```sql
ALTER TABLE continuous_experiments
MODIFY COLUMN target_duration_weeks FLOAT NOT NULL;
```

**Migration Script**: [scripts/migrate_target_duration_to_float.py](../scripts/migrate_target_duration_to_float.py)

**Usage**:
```bash
# Dry run (preview changes)
python scripts/migrate_target_duration_to_float.py --dry-run

# Apply migration
python scripts/migrate_target_duration_to_float.py
```

### 2. Application Updates ✓

#### Backend ([web/app.py](../web/app.py))
- Updated API documentation to indicate `target_duration_weeks: float`
- Added validation to ensure value is between 0 and 1000
- Added proper error messages for invalid values

#### Orchestrator ([src/continuous/recording_orchestrator.py](../src/continuous/recording_orchestrator.py))
- Added explicit `float()` conversion for target_duration_weeks
- Added logging to show calculated total cycles and minutes
- Now supports fractional weeks correctly

#### Frontend ([web/templates/continuous_experiment.html](../web/templates/continuous_experiment.html))
- Already configured correctly with `step="0.01"` on the input field
- Help text explains fractional weeks (e.g., 0.01 = ~10 minutes)

## Creating a New Test Experiment

### Using the Web Interface

1. **Start the web server** (if not already running):
   ```bash
   source /Users/bernd/miniconda3/bin/activate xorProject
   cd /Users/bernd/python/XOR
   python web/app.py
   ```

2. **Navigate to**: http://localhost:5000/continuous

3. **Click**: "New Experiment" button

4. **Fill in the form**:
   - **Experiment Name**: "Test Fractional Weeks - 0.01"
   - **Duration (weeks)**: `0.01` (will run for ~1.6 hours)
   - **Recording Interval**: `6` minutes
   - **Recording Duration**: `180` seconds (3 minutes)
   - **Playback File**: Select from dropdown
   - **Channel 1 (Left)**: `lavendar` (class 1)
   - **Channel 2 (Right)**: `empty` (class 0)

5. **Click**: "Create Experiment"

6. **Start the experiment**: Click the "Start" button

### Expected Behavior with 0.01 Weeks

```
Target duration: 0.01 weeks
Recording interval: 6 minutes

Calculation:
  Total minutes = 0.01 × 7 × 24 × 60 = 100.8 minutes
  Total cycles = 100.8 ÷ 6 = 16.8 → 16 cycles (integer)
  Actual duration = 16 × 6 = 96 minutes (1.6 hours)
```

Each cycle:
- Start recording (3 minutes)
- Process and run Auto-QC
- Sleep for remaining time until next 6-minute mark
- Total: ~6 minutes per cycle

**Total experiment duration**: ~96 minutes (16 cycles × 6 minutes)

### Monitoring Progress

Watch the experiment in real-time:

1. **Web Dashboard**: http://localhost:5000/continuous
   - Shows current cycle progress
   - Displays QC pass/fail counts
   - Shows current accuracy

2. **Log File**:
   ```bash
   tail -f logs/continuous/<experiment_id>.log
   ```

3. **Database Query**:
   ```bash
   source /Users/bernd/miniconda3/bin/activate xorProject
   python -c "
   from src.db_connection import DatabaseConnection
   db = DatabaseConnection(backend='mariadb')
   with db.get_connection() as conn:
       cursor = conn.cursor(dictionary=True)
       cursor.execute('''
           SELECT experiment_id, experiment_name, status,
                  current_cycle, total_cycles_expected,
                  qc_pass_count, qc_fail_count
           FROM continuous_experiments
           WHERE status = 'running'
       ''')
       for exp in cursor.fetchall():
           print(f\"{exp['experiment_id']}: {exp['current_cycle']}/{exp['total_cycles_expected']} cycles\")
   "
   ```

## Verification of Fix

### Before Migration (Bug):
- Input: `0.01` weeks
- Stored: `0` (INT truncation)
- Cycles: `0`
- Duration: 0 seconds (immediate completion)

### After Migration (Fixed):
- Input: `0.01` weeks
- Stored: `0.01` (FLOAT preserved)
- Cycles: `16`
- Duration: ~96 minutes

## Testing Different Durations

Common test durations:

| Weeks | Minutes | Hours | Cycles (6min) | Description |
|-------|---------|-------|---------------|-------------|
| 0.001 | 10.08   | 0.17  | 1             | Quick smoke test |
| 0.01  | 100.8   | 1.68  | 16            | Short test run |
| 0.1   | 1008    | 16.8  | 168           | Extended test |
| 1.0   | 10080   | 168   | 1680          | Full week |

## Rollback (if needed)

If you need to revert to INT (not recommended):

```sql
ALTER TABLE continuous_experiments
MODIFY COLUMN target_duration_weeks INT NOT NULL;
```

**Warning**: This will truncate any existing fractional values!

## Related Files

- Migration script: [scripts/migrate_target_duration_to_float.py](../scripts/migrate_target_duration_to_float.py)
- Web application: [web/app.py](../web/app.py) (lines 995, 1024-1032)
- Orchestrator: [src/continuous/recording_orchestrator.py](../src/continuous/recording_orchestrator.py) (lines 535-540)
- Frontend: [web/templates/continuous_experiment.html](../web/templates/continuous_experiment.html) (line 106)

## Future Improvements

Consider adding:
1. **Duration presets** in UI: "10 minutes", "1 hour", "1 day", "1 week"
2. **Duration calculator** showing exact cycles and runtime
3. **Validation warning** if duration is too short (< 0.001 weeks / 10 minutes)
4. **Confirmation dialog** for very long durations (> 4 weeks)
