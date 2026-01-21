# Experiment Re-run Feature

## Overview

The web interface now includes a **Re-run (Duplicate) Experiment** feature that allows you to easily create new experiments based on existing ones with automatic name increment.

## What It Does

When you re-run an experiment:

1. **Copies Configuration**: All settings from the source experiment are copied:
   - Target duration (weeks)
   - Recording interval and duration
   - Playback file
   - Channel substances (channel 1 & 2)
   - Beaker configuration
   - Auto-QC thresholds
   - Training parameters (batch size, epochs, sliding window)
   - All other experimental settings

2. **Auto-increments Name**: The experiment name is automatically updated:
   - `"Lavender Test"` → `"Lavender Test (2)"`
   - `"Lavender Test (2)"` → `"Lavender Test (3)"`
   - Ensures uniqueness by checking the database

3. **Creates Fresh Experiment**:
   - New experiment ID is generated
   - Status starts as "stopped" (ready to start)
   - No old data is copied (cycles, alerts, etc.)
   - Progress starts from 0

4. **Optional Auto-start**: You can choose to start the experiment immediately or create it in stopped state

## How to Use

### Option 1: From Experiment Card (Dashboard)

1. Navigate to the Continuous Learning Dashboard at `/continuous`
2. Find the experiment you want to re-run
3. Click the **"Re-run"** button in the card footer
4. A confirmation dialog will ask:
   - "Would you like to start it immediately?" → Click **OK** to start now, or **Cancel** to continue
   - If you clicked Cancel: "Create the experiment without starting it?" → Click **OK** to create stopped experiment

### Option 2: From Experiment Details Modal

1. Click on any experiment card to open the details modal
2. At the bottom of the modal, click the **"Re-run"** button (always visible)
3. Follow the same confirmation prompts as above

## API Endpoint

**Endpoint:** `POST /api/continuous/experiments/<experiment_id>/duplicate`

**Request Body (optional):**
```json
{
  "experiment_name": "Custom Name (optional)",
  "auto_start": true  // false by default
}
```

**Response:**
```json
{
  "success": true,
  "experiment_id": "exp_abc123ef",
  "experiment_name": "Lavender Test (2)",
  "message": "Experiment duplicated successfully as \"Lavender Test (2)\"",
  "source_experiment_id": "exp_original",
  "started": false  // true if auto_start was requested
}
```

## Use Cases

### 1. Replicate Successful Experiments
Re-run an experiment that produced good results to verify reproducibility:
```
Original: "Lavender vs Empty - 2 Week Run"
Re-run:   "Lavender vs Empty - 2 Week Run (2)"
```

### 2. Quick Testing with Same Parameters
Create multiple test runs with identical configuration:
```
Test 1: "Quick Test (10 min)"
Test 2: "Quick Test (10 min) (2)"
Test 3: "Quick Test (10 min) (3)"
```

### 3. Long-term Studies
Run repeated experiments over time with consistent settings:
```
Week 1: "Baseline Study"
Week 2: "Baseline Study (2)"
Week 3: "Baseline Study (3)"
```

### 4. Before/After Comparisons
Test environmental changes (e.g., Faraday cage on/off) by creating duplicate experiments and modifying one parameter via the create form.

## Name Increment Logic

The system intelligently handles experiment names:

1. **No run number**: `"Experiment A"` → `"Experiment A (2)"`
2. **Existing run number**: `"Experiment A (2)"` → `"Experiment A (3)"`
3. **Name collision**: If `"Experiment A (2)"` already exists, it tries `(3)`, `(4)`, etc. until unique
4. **Manual override**: You can provide a custom name via API (not available in UI)

## Implementation Details

### Backend (Flask Route)

**File:** `web/app.py`

**Route:** `@app.route('/api/continuous/experiments/<experiment_id>/duplicate', methods=['POST'])`

**Key features:**
- Fetches source experiment from database
- Parses experiment name for run numbers using regex
- Ensures name uniqueness
- Creates new experiment record with fresh ID
- Optionally starts the orchestrator process
- Logs all actions for audit trail

### Frontend (JavaScript)

**File:** `web/templates/continuous_dashboard.html`

**Function:** `async function rerunExperiment(experimentId, experimentName)`

**Key features:**
- User-friendly confirmation dialogs
- Two-step confirmation (start immediately vs. create stopped)
- Automatic dashboard refresh after creation
- Error handling with clear messages
- Modal auto-close after successful creation

### UI Components

1. **Card Footer Button**: Small "Re-run" button in each experiment card
2. **Modal Footer Button**: Prominent "Re-run" button in experiment details modal
3. **Both buttons call the same `rerunExperiment()` function**

## Safety Features

- ✅ **No data loss**: Original experiment remains unchanged
- ✅ **Name uniqueness**: Automatically prevents duplicate names
- ✅ **User confirmation**: Requires explicit user action
- ✅ **Audit trail**: All duplications are logged
- ✅ **Process isolation**: New experiment runs independently

## Limitations

- Cannot modify configuration during re-run (must edit after creation if changes needed)
- Custom names must be provided via API (not available in UI dialog)
- Re-run button is always visible (even for running experiments)

## Future Enhancements

Possible improvements for future versions:

1. **Pre-edit dialog**: Allow user to modify settings before creating
2. **Batch re-run**: Create multiple duplicates at once
3. **Template system**: Save experiment configurations as templates
4. **Smart naming**: Suggest names based on date/time (e.g., "Exp A - Jan 2026")
5. **Copy with modifications**: Quick edit dialog for common changes (duration, substances)

## Testing

To test the re-run feature:

```bash
# 1. Start the web interface
python web/app.py

# 2. Navigate to http://localhost:5001/continuous

# 3. Create a test experiment via the "New Experiment" button

# 4. Click "Re-run" on the created experiment

# 5. Verify the new experiment appears with incremented name

# 6. Check logs to confirm successful duplication:
tail -f logs/continuous/*.log
```

## Troubleshooting

### Issue: "Failed to duplicate experiment: Source experiment not found"
**Solution**: Ensure the experiment ID exists in the database

### Issue: Name keeps incrementing unexpectedly (e.g., jumps to (5))
**Solution**: This is normal - it means experiments with numbers (2), (3), (4) already exist. The system finds the next available number.

### Issue: Re-run button doesn't respond
**Solution**: Check browser console for JavaScript errors. Ensure the modal is fully loaded before clicking.

### Issue: Experiment created but not started
**Solution**: Check the orchestrator logs. The experiment may have been created successfully but failed to start due to missing dependencies or configuration issues.

## Related Files

- Backend: [`web/app.py`](../web/app.py) (lines 1435-1605)
- Frontend: [`web/templates/continuous_dashboard.html`](../web/templates/continuous_dashboard.html) (lines 334-336, 552-556, 870-911)
- Database: `continuous_experiments` table in MariaDB
- Documentation: [`docs/WEB_INTERFACE_GUIDE.md`](WEB_INTERFACE_GUIDE.md)

## API Examples

### Basic Re-run (Create Stopped)

```bash
curl -X POST http://localhost:5001/api/continuous/experiments/exp_abc123/duplicate \
  -H "Content-Type: application/json"
```

### Re-run and Auto-start

```bash
curl -X POST http://localhost:5001/api/continuous/experiments/exp_abc123/duplicate \
  -H "Content-Type: application/json" \
  -d '{"auto_start": true}'
```

### Re-run with Custom Name

```bash
curl -X POST http://localhost:5001/api/continuous/experiments/exp_abc123/duplicate \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "My Custom Experiment Name", "auto_start": false}'
```

---

**Last Updated:** 2026-01-21
**Author:** Claude Code
**Version:** 1.0
