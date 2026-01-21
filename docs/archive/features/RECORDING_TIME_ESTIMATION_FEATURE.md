# Recording Time Estimation Feature

## Overview

Added recording time estimation and progress visualization to the web interface. When a recording is in progress, users can now see:
- **Elapsed time** - How long the recording has been running
- **Remaining time** - How much time is left until completion
- **Progress bar** - Visual representation of recording progress
- **Total duration** - The configured recording duration

## Changes Made

### 1. Backend API Enhancement ([web/app.py](web/app.py))

**Modified endpoint:** `/api/rubix44/status`

Added time estimation logic that calculates and formats recording progress:

```python
@app.route('/api/rubix44/status')
def api_rubix44_status():
    """
    API endpoint: Get rubix44 recording status with time estimation
    """
    # ... get status from rubix44 ...

    # Add time estimation if recording is active
    if status.get('status') == 'recording':
        elapsed = status.get('elapsed_seconds', 0)
        duration = status.get('duration_seconds', 0)

        if duration > 0:
            # Calculate remaining time
            remaining = max(0, duration - elapsed)
            progress_percent = (elapsed / duration) * 100

            # Add estimation fields
            status['time_remaining_seconds'] = remaining
            status['progress_percent'] = progress_percent

            # Format time strings for display
            status['elapsed_formatted'] = format_duration(elapsed)
            status['remaining_formatted'] = format_duration(remaining)
            status['duration_formatted'] = format_duration(duration)
```

**New helper function:** `format_duration(seconds)`

Formats seconds into human-readable strings with appropriate units:
- `30s` → "30s" (under 1 minute)
- `90s` → "1m 30s" (under 1 hour)
- `3600s` → "1h 0m" (under 24 hours)
- `5430s` → "1h 30m" (under 24 hours)
- `8640s` → "2h 24m" (0.1 days shown as hours)
- `86400s` → "1d" (exactly 1 day)
- `90000s` → "1d 1h" (more than 1 day)

### 2. Frontend UI Enhancement ([web/templates/annotate.html](web/templates/annotate.html))

**Modified function:** `checkRecordingStatus()`

Updated the recording status display to show:

```javascript
// Show time information if available
if (data.elapsed_formatted && data.remaining_formatted && data.duration_formatted) {
    statusHtml += '<div class="mt-2">';
    statusHtml += `<div class="d-flex justify-content-between mb-1">`;
    statusHtml += `<span><i class="bi bi-clock-history"></i> Elapsed: <strong>${data.elapsed_formatted}</strong></span>`;
    statusHtml += `<span><i class="bi bi-hourglass-split"></i> Remaining: <strong>${data.remaining_formatted}</strong></span>`;
    statusHtml += `</div>`;
    statusHtml += `<small class="text-muted">Total duration: ${data.duration_formatted}</small>`;
    statusHtml += '</div>';

    // Progress bar
    if (data.progress_percent !== undefined) {
        const percent = Math.min(100, Math.max(0, data.progress_percent)).toFixed(1);
        statusHtml += '<div class="progress mt-2" style="height: 25px;">';
        statusHtml += `<div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" `;
        statusHtml += `role="progressbar" style="width: ${percent}%" ...>`;
        statusHtml += `${percent}%`;
        statusHtml += '</div></div>';
    }
}
```

## API Response Format

When a recording is active, the `/api/rubix44/status` endpoint now returns:

```json
{
  "status": "recording",
  "session_id": "rec_20260112_143022",
  "elapsed_seconds": 1800,
  "duration_seconds": 3600,
  "time_remaining_seconds": 1800,
  "progress_percent": 50.0,
  "elapsed_formatted": "30m 0s",
  "remaining_formatted": "30m 0s",
  "duration_formatted": "1h 0m"
}
```

## Visual Display

The recording status display now shows:

```
┌─────────────────────────────────────────┐
│ 🔴 Recording in progress...             │
│ Session: rec_20260112_143022            │
│                                         │
│ 🕐 Elapsed: 30m 0s   ⏳ Remaining: 30m 0s│
│ Total duration: 1h 0m                   │
│                                         │
│ ████████████░░░░░░░░░░░░ 50.0%         │
│                                         │
│ ⌛ Loading...                           │
└─────────────────────────────────────────┘
```

## Technical Details

### Status Update Frequency
- The frontend polls the status endpoint every **5 seconds**
- Progress bar updates automatically on each poll
- Smooth animated progress bar provides visual feedback

### Backward Compatibility
- Falls back gracefully if `elapsed_seconds` or `duration_seconds` are not provided by rubix44
- Shows basic elapsed time if formatted strings are unavailable

### Progress Bar Features
- **Striped animation** - Provides visual indication that recording is active
- **Warning color** (yellow/orange) - Indicates recording in progress
- **Percentage display** - Shows exact progress within the bar
- **Bounded values** - Ensures progress stays between 0% and 100%

## Testing

To test the feature:

1. Start the web server:
   ```bash
   python web/app.py --port 5001
   ```

2. Navigate to the annotation page:
   ```
   http://localhost:5001/annotate
   ```

3. Start a new recording (make sure rubix44 server is running)

4. Observe the "Recording Status" panel showing:
   - Real-time elapsed time
   - Calculated remaining time
   - Animated progress bar
   - Formatted durations

## Future Enhancements

Potential improvements:
- Add estimated completion timestamp (e.g., "Will finish at 15:45")
- Add notification when recording is about to complete (e.g., "1 minute remaining")
- Add estimated time to other pages (dashboard, continuous learning experiments)
- Add pause/resume functionality with time tracking
- Store time estimation history in database for analysis

## Dependencies

No new dependencies required. Uses:
- Existing rubix44 API client
- Bootstrap progress bars (already included)
- Bootstrap icons (already included)

## Files Modified

1. **[web/app.py](web/app.py:535-583)** - Backend API endpoint and time formatting
2. **[web/templates/annotate.html](web/templates/annotate.html:694-762)** - Frontend status display

## Compatibility

- **rubix44-recorder API v1.1.0+** - Requires `elapsed_seconds` and `duration_seconds` fields
- Works with existing continuous learning experiments
- No database schema changes required
