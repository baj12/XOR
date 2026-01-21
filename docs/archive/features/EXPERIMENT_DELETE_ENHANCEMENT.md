# Experiment Delete Enhancement

## Overview

Enhanced the experiment deletion functionality to completely remove **all** related data from the database and disk, not just the experiment record itself.

## Problem

The original delete function only removed:
- ❌ Experiment record from `continuous_experiments`
- ❌ Alerts from `experiment_alerts`
- ❌ Cycles from `recording_cycles`

But it **left behind**:
- ⚠️ Recording sessions in `recording_sessions` table
- ⚠️ Extracted features in `features` table
- ⚠️ Model files on disk (`models/continuous/{experiment_id}/`)

This meant the database would accumulate orphaned data over time.

## Solution

Updated the delete endpoint ([web/app.py:1425-1544](web/app.py:1425-1544)) to perform **complete cleanup**:

### Backend Changes

```python
@app.route('/api/continuous/experiments/<experiment_id>', methods=['DELETE'])
def api_continuous_experiment_delete(experiment_id):
    """
    Deletes:
    - Experiment record from continuous_experiments
    - All related alerts from experiment_alerts
    - All related cycles from recording_cycles
    - All related recording sessions from recording_sessions  # NEW
    - All extracted features from features table             # NEW
    - Model files from disk                                  # NEW
    """
```

### Deletion Sequence

1. **Safety check** - Cannot delete running experiments
2. **Get session IDs** - Find all sessions associated with this experiment
3. **Count records** - Track what will be deleted for reporting
4. **Database cleanup**:
   - Delete `experiment_alerts`
   - Delete `recording_cycles`
   - Delete `features` for all sessions (NEW)
   - Delete `recording_sessions` (NEW)
   - Delete `continuous_experiments`
5. **Disk cleanup** - Remove model directory (NEW)

### Code Example

```python
# Get all session IDs for this experiment
cursor.execute("""
    SELECT session_id FROM recording_cycles
    WHERE experiment_id = %s AND session_id IS NOT NULL
""", (experiment_id,))
session_ids = [row['session_id'] for row in cursor.fetchall()]

# Delete features for all sessions
if session_ids:
    placeholders = ','.join(['%s'] * len(session_ids))
    cursor.execute(f"DELETE FROM features WHERE session_id IN ({placeholders})", session_ids)

# Delete recording sessions
cursor.execute("DELETE FROM recording_sessions WHERE experiment_id = %s", (experiment_id,))

# Delete model files from disk
model_dir = Path(f"models/continuous/{experiment_id}")
if model_dir.exists():
    shutil.rmtree(model_dir)
```

## Frontend Changes

Updated the delete confirmation and success messages to show all deletions:

### Before
```javascript
alert(
    `✓ Experiment deleted successfully!\n\n` +
    `Removed:\n` +
    `  • ${data.deleted.cycles} recording cycles\n` +
    `  • ${data.deleted.alerts} alerts`
);
```

### After
```javascript
alert(
    `✓ Experiment deleted successfully!\n\n` +
    `Removed:\n` +
    `  • Experiment: ${deleted.experiment}\n` +
    `  • Cycles: ${deleted.cycles}\n` +
    `  • Alerts: ${deleted.alerts}\n` +
    `  • Recording sessions: ${deleted.recording_sessions}\n` +
    `  • Features: ${deleted.features}\n` +
    `  • Model files: Yes/No`
);
```

## API Response Format

The delete endpoint now returns detailed counts:

```json
{
  "success": true,
  "message": "Experiment 'my_experiment' deleted successfully",
  "deleted": {
    "experiment": "my_experiment",
    "cycles": 150,
    "alerts": 5,
    "recording_sessions": 150,
    "features": 150000,
    "model_files": 1
  }
}
```

## Safety Features

### 1. Cannot Delete Running Experiments
```python
if experiment['status'] == 'running':
    return jsonify({
        'success': False,
        'error': 'Cannot delete running experiment. Stop or kill it first.'
    }), 400
```

### 2. Double Confirmation
User must confirm deletion with a warning about permanent data loss.

### 3. Transaction Safety
All database operations are in a transaction - if any step fails, all changes are rolled back.

### 4. Graceful Disk Cleanup
If model directory deletion fails, it logs a warning but doesn't fail the entire operation.

```python
try:
    shutil.rmtree(model_dir)
except Exception as e:
    logger.warning(f"Failed to delete model directory: {e}")
    # Continue - database cleanup still succeeded
```

## Testing the Fix

### 1. Create a Test Experiment
```bash
# Via web interface at http://localhost:5001/continuous/create
# Or via API
```

### 2. Let it Run for a Few Cycles
This will create:
- Recording cycles
- Recording sessions
- Extracted features in database
- Model files on disk

### 3. Stop the Experiment
Click "Pause" or "Force Kill" to stop it.

### 4. Delete the Experiment
1. Click the experiment card
2. Click "Delete" button
3. Confirm the warning dialog
4. See detailed deletion report

### 5. Verify Cleanup

**Check database:**
```sql
-- Should return 0 rows
SELECT * FROM recording_sessions WHERE experiment_id = 'exp_abc123';
SELECT * FROM features WHERE session_id IN (SELECT session_id FROM recording_cycles WHERE experiment_id = 'exp_abc123');
SELECT * FROM recording_cycles WHERE experiment_id = 'exp_abc123';
SELECT * FROM experiment_alerts WHERE experiment_id = 'exp_abc123';
SELECT * FROM continuous_experiments WHERE experiment_id = 'exp_abc123';
```

**Check disk:**
```bash
# Should not exist
ls -la models/continuous/exp_abc123/
```

## Database Impact

### Tables Modified
- `features` - Removes orphaned feature data
- `recording_sessions` - Removes session metadata
- `recording_cycles` - Removes cycle history
- `experiment_alerts` - Removes alert history
- `continuous_experiments` - Removes experiment record

### Storage Reclaimed

For a typical experiment with 100 cycles:
- **Features table**: ~100k rows (~50-100 MB)
- **Recording sessions**: 100 rows (~10 KB)
- **Recording cycles**: 100 rows (~50 KB)
- **Model files on disk**: ~50-500 MB depending on architecture

**Total: ~100-600 MB per experiment**

## Benefits

✅ **No orphaned data** - Complete cleanup
✅ **Reclaimed storage** - Database and disk space freed
✅ **Faster queries** - Smaller tables perform better
✅ **Clear audit trail** - Detailed deletion report
✅ **Safe operation** - Cannot delete running experiments
✅ **User transparency** - Shows exactly what was deleted

## Migration for Existing Orphaned Data

If you have existing orphaned data from experiments deleted with the old code:

```sql
-- Find orphaned recording sessions (no matching experiment)
SELECT rs.* FROM recording_sessions rs
LEFT JOIN continuous_experiments ce ON rs.experiment_id = ce.experiment_id
WHERE ce.experiment_id IS NULL;

-- Find orphaned features (no matching cycle)
SELECT f.* FROM features f
LEFT JOIN recording_cycles rc ON f.session_id = rc.session_id
WHERE rc.session_id IS NULL;

-- Clean up orphaned data (BE CAREFUL!)
-- DELETE FROM recording_sessions WHERE experiment_id NOT IN (SELECT experiment_id FROM continuous_experiments);
-- DELETE FROM features WHERE session_id NOT IN (SELECT session_id FROM recording_cycles WHERE session_id IS NOT NULL);
```

## Files Modified

1. **[web/app.py:1425-1544](web/app.py:1425-1544)** - Backend delete endpoint
2. **[web/templates/continuous_dashboard.html:554-599](web/templates/continuous_dashboard.html:554-599)** - Frontend delete UI

## Conclusion

The experiment deletion now performs **complete cleanup** of all related data, preventing database bloat and making it safe to create/delete experiments during testing and development.
