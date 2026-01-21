# Foreign Key Constraint Fix

## Error

```
ERROR rubix44_error
Failed to start recording: 1452 (23000): Cannot add or update a child row:
a foreign key constraint fails (`xor_project`.`recording_cycles`,
CONSTRAINT `recording_cycles_ibfk_2` FOREIGN KEY (`session_id`)
REFERENCES `recording_sessions` (`session_id`) ON DELETE SET NULL)
```

## Root Cause

### The Problem

The code was trying to update `recording_cycles.session_id` **before** creating the corresponding entry in `recording_sessions`, violating the foreign key constraint.

### Database Constraint

```sql
ALTER TABLE recording_cycles
ADD CONSTRAINT recording_cycles_ibfk_2
FOREIGN KEY (session_id)
REFERENCES recording_sessions(session_id)
ON DELETE SET NULL;
```

This constraint ensures that any `session_id` in `recording_cycles` must first exist in `recording_sessions`.

### Original Flow (BROKEN)

```
1. create_recording_cycle(1)
   → Creates recording_cycles row with session_id = NULL

2. start_recording_cycle(1)
   → Calls Rubix44 API
   → Gets session_id = "20260114_175618"
   → update_cycle_status(1, 'recording', session_id="20260114_175618")
      ❌ ERROR: recording_sessions doesn't have this session_id yet!

3. create_recording_metadata("20260114_175618", 1)
   → Would create recording_sessions entry
   → But never reached due to error in step 2
```

## The Fix

### Solution

**Move the `session_id` update to AFTER `recording_sessions` is created.**

### New Flow (FIXED)

```
1. create_recording_cycle(1)
   → Creates recording_cycles row with session_id = NULL

2. start_recording_cycle(1)
   → Calls Rubix44 API
   → Gets session_id = "20260114_175618"
   → update_cycle_status(1, 'recording')  ✓ No session_id yet

3. wait_for_recording_completion("20260114_175618", 1)
   → Polls until recording completes

4. create_recording_metadata("20260114_175618", 1)
   → Creates recording_sessions entry with session_id ✓
   → Now updates recording_cycles with session_id ✓ SAFE - FK satisfied!
```

### Code Changes

#### Change 1: Don't set session_id immediately

**File**: `src/continuous/recording_orchestrator.py:214-217`

**Before**:
```python
self.logger.info(f"Recording started: {session_id}")
self.update_cycle_status(cycle_number, 'recording', session_id=session_id)
```

**After**:
```python
self.logger.info(f"Recording started: {session_id} ({human_id})")
self.logger.info(f"  Status: {status}, Duration: {duration}s")

# NOTE: Don't update cycle with session_id yet - recording_sessions entry
# must be created first due to foreign key constraint.
# Session ID will be set in create_recording_metadata()
self.update_cycle_status(cycle_number, 'recording')
```

#### Change 2: Set session_id after creating recording_sessions

**File**: `src/continuous/recording_orchestrator.py:399-408`

**After `recording_sessions` INSERT/UPDATE**:
```python
conn.commit()
self.logger.info("Metadata created successfully")

# Now that recording_sessions entry exists, we can safely update
# recording_cycles with the session_id (satisfies foreign key constraint)
# Note: We don't change the status here, just add the session_id
cursor.execute("""
    UPDATE recording_cycles
    SET session_id = %s
    WHERE experiment_id = %s AND cycle_number = %s
""", (session_id, self.experiment_id, cycle_number))
conn.commit()
self.logger.debug(f"Updated cycle {cycle_number} with session_id {session_id}")
```

## Why This Happened

This issue surfaced **after** fixing the Rubix44 API response parsing because:

1. **Before the fix**: `start_recording_cycle()` always returned `None` (couldn't parse session_id)
2. **After the fix**: `start_recording_cycle()` correctly returns the session_id
3. **Result**: Code that was never executed before (setting session_id) now runs and hits the FK constraint

## Database Schema

### recording_sessions (parent table)

```sql
CREATE TABLE recording_sessions (
    session_id VARCHAR(255) PRIMARY KEY,  -- Must exist first
    experiment_id VARCHAR(100),
    cycle_number INT,
    recording_date DATETIME,
    ...
);
```

### recording_cycles (child table)

```sql
CREATE TABLE recording_cycles (
    cycle_id INT PRIMARY KEY AUTO_INCREMENT,
    experiment_id VARCHAR(100),
    cycle_number INT,
    session_id VARCHAR(100),  -- Foreign key to recording_sessions
    ...
    FOREIGN KEY (session_id) REFERENCES recording_sessions(session_id)
);
```

## Testing

### Before Fix

```bash
# Start experiment
python -m src.continuous.orchestrator ...

# Result:
ERROR - Failed to start recording: 1452 (23000): Cannot add or update a child row...
```

### After Fix

```bash
# Start experiment
python -m src.continuous.orchestrator ...

# Result:
INFO - Recording started: 20260114_175618 (swift-panda-2347)
INFO - Waiting for recording to complete...
INFO - Metadata created successfully
DEBUG - Updated cycle 1 with session_id 20260114_175618
INFO - Recording completed successfully
```

## Verification Queries

### Check if session exists in both tables

```sql
-- Check recording_sessions
SELECT session_id, experiment_id, cycle_number
FROM recording_sessions
WHERE session_id = '20260114_175618';

-- Check recording_cycles
SELECT cycle_id, experiment_id, cycle_number, session_id
FROM recording_cycles
WHERE session_id = '20260114_175618';

-- Both should return the same session_id
```

### Check foreign key integrity

```sql
-- This query should return no rows (all session_ids have valid references)
SELECT rc.cycle_id, rc.session_id
FROM recording_cycles rc
LEFT JOIN recording_sessions rs ON rc.session_id = rs.session_id
WHERE rc.session_id IS NOT NULL
  AND rs.session_id IS NULL;
```

## Alternative Solutions Considered

### Option 1: Remove Foreign Key Constraint

**Pros**: Simple, no code changes needed
**Cons**: Loses referential integrity, can have orphaned session_ids

### Option 2: Make Foreign Key DEFERRED

**Pros**: Constraint checked at transaction end
**Cons**: Not supported in MySQL/MariaDB (PostgreSQL only)

### Option 3: Create recording_sessions first, then recording_cycles

**Pros**: Clean order
**Cons**: Requires restructuring the entire flow, can't create recording_sessions without a session_id

### Option 4: Update session_id after metadata creation ✅ CHOSEN

**Pros**:
- Minimal code changes
- Maintains referential integrity
- Clear execution order
- Easy to understand and maintain

**Cons**: None

## Related Issues

- **Rubix44 API Fix**: [RUBIX44_API_FIX.md](RUBIX44_API_FIX.md)
- **Enhancements**: [RUBIX44_ENHANCEMENTS_SUMMARY.md](RUBIX44_ENHANCEMENTS_SUMMARY.md)

## Impact

- ✅ Fixes all future recordings
- ✅ No database schema changes needed
- ✅ Maintains referential integrity
- ✅ Clear code comments explain the constraint
- ✅ No impact on existing data

## Conclusion

The foreign key constraint error was caused by attempting to set `recording_cycles.session_id` before the corresponding `recording_sessions` entry existed. The fix ensures the proper order: create the parent record first, then update the child record with the foreign key reference.

This is a **correct fix** that maintains database integrity while allowing the recording workflow to proceed successfully.
