# ⚠️ RUBIX44 SERVER CHANGES REQUIRED

**Date:** 2026-01-04
**Priority:** HIGH - Required for proper integration

## Overview

The following changes are needed on the rubix44-recorder server to fully support the recording management system workflow.

## Required API Enhancements

### 1. Stop Recording Endpoint (CRITICAL)

**Current Issue:** No way to stop an in-progress recording from the web interface.

**Required Endpoint:**
```
POST /api/v1/recordings/stop
```

**Expected Behavior:**
- Stop the current recording immediately
- Finalize and save the WAV files
- Return the session ID and file paths

**Expected Response:**
```json
{
  "success": true,
  "session_id": "recording_2026-01-04_16-30-00",
  "files": [
    {
      "name": "recording_2026-01-04_16-30-00_stereo.wav",
      "path": "/path/to/file",
      "size": 1234567
    }
  ],
  "duration_seconds": 3625.5
}
```

**Implementation Notes:**
- Should work even if recording hasn't reached the specified duration
- Should calculate actual duration from start time to stop time
- Should update the recording history with actual duration

### 2. Recording Status Enhancements

**Current Issue:** Status endpoint doesn't provide enough detail about the current recording.

**Required Fields in `/api/v1/recordings/status`:**
```json
{
  "status": "recording",  // or "idle"
  "session_id": "recording_2026-01-04_16-30-00",
  "start_time": "2026-01-04T16:30:00",
  "elapsed_seconds": 125.5,
  "playback_file": "noise_baseline.wav",  // Currently playing
  "output_prefix": "recording",
  "expected_duration": 3600
}
```

**Currently Missing:**
- `start_time`: When recording started (ISO format)
- `elapsed_seconds`: How long has been recording
- `playback_file`: Which file is currently playing back
- `output_prefix`: The prefix being used

**Use Case:**
- Web interface can display: "Recording in progress: 2:05 / 60:00"
- Can show which playback file is active
- Can update duration field automatically when recording is selected

### 3. Recording History Enhancements

**Current Issue:** History doesn't include all metadata needed for annotation.

**Required Fields in `/api/v1/recordings/history`:**

Each recording should include:
```json
{
  "id": "recording_2026-01-04_16-30-00",
  "prefix": "recording",
  "timestamp": "2026-01-04_16-30-00",
  "start_time": "2026-01-04T16:30:00",  // NEW: ISO format
  "end_time": "2026-01-04T17:30:25",    // NEW: ISO format
  "duration_seconds": 3625.5,            // NEW: Actual duration
  "playback_file": "noise_baseline.wav", // NEW: What was played
  "sample_rate": 44100,                  // NEW: Audio sample rate
  "files": [
    {
      "name": "recording_2026-01-04_16-30-00_stereo.wav",
      "path": "/full/path/to/file",
      "size": 123456789,
      "modified": "2026-01-04T17:30:25"
    }
  ]
}
```

**Currently Missing:**
- `start_time`: Recording start timestamp
- `end_time`: Recording end timestamp
- `duration_seconds`: Actual recording duration
- `playback_file`: Which playback file was used
- `sample_rate`: Audio configuration

**Use Case:**
- Automatically populate duration when recording is selected for annotation
- Track which playback file was used (important for metadata)
- Show actual vs expected duration

### 4. Playback File Metadata

**Current Issue:** Playback files endpoint only returns filenames.

**Required Enhancement for `/api/v1/playback-files`:**
```json
[
  {
    "filename": "noise_baseline.wav",
    "path": "/full/path/to/noise_baseline.wav",
    "size": 987654321,
    "duration_seconds": 120.5,  // NEW: Duration of the file
    "sample_rate": 44100,        // NEW: Sample rate
    "channels": 2,               // NEW: Mono/stereo
    "format": "WAV",             // NEW: File format
    "modified": "2026-01-01T12:00:00"
  }
]
```

**Currently Missing:**
- Audio file metadata (duration, sample rate, channels)
- Full file information

**Use Case:**
- Validate recording settings before starting
- Show expected recording size
- Prevent configuration mismatches

## Optional Enhancements

### 5. Recording Progress Callback (Nice to Have)

**Feature:** WebSocket or SSE endpoint for real-time progress updates.

**Endpoint:**
```
GET /api/v1/recordings/progress (Server-Sent Events)
```

**Event Stream:**
```
event: progress
data: {"elapsed": 125.5, "status": "recording"}

event: complete
data: {"session_id": "...", "duration": 3625.5}

event: error
data: {"error": "Disk space low"}
```

**Use Case:**
- Live progress bar in web interface
- Real-time recording status
- Better user experience

### 6. Delete Recording Endpoint (Nice to Have)

**Endpoint:**
```
DELETE /api/v1/recordings/<session_id>
```

**Use Case:**
- Clean up failed or test recordings
- Manage disk space
- Remove recordings rejected during QC

## Implementation Priority

### P0 - Critical (Required for MVP)
1. ✅ Stop recording endpoint
2. ✅ Recording status enhancements (elapsed time, playback file)
3. ✅ History metadata (duration, playback file)

### P1 - Important (Needed for good UX)
4. ⚠️ Playback file metadata
5. ⚠️ Start/end timestamps in history

### P2 - Nice to Have
6. ⏸️ Progress updates (WebSocket/SSE)
7. ⏸️ Delete recording endpoint

## Testing Checklist

After implementing changes, test:

- [ ] Stop recording via API works
- [ ] Status shows elapsed time during recording
- [ ] Status shows playback file being used
- [ ] History includes duration_seconds
- [ ] History includes playback_file
- [ ] History includes start_time/end_time
- [ ] Web interface can fetch and display all new fields

## API Compatibility

**Backward Compatibility:**
- All new fields should be ADDED, not replacing existing ones
- Existing fields should maintain current format
- Clients without updates should continue to work

**Version Consideration:**
- Consider adding `/api/v2/` endpoints if breaking changes needed
- Document API version in response headers

## Example Integration Code

### Stop Recording (Web Interface)
```javascript
// In web/templates/annotate.html
async function stopRecording() {
    const response = await fetch('http://10.0.0.58:5000/api/v1/recordings/stop', {
        method: 'POST'
    });
    const result = await response.json();

    if (result.success) {
        // Use actual duration from response
        console.log(`Recording stopped: ${result.duration_seconds}s`);
        // Refresh recordings list
        loadRubix44Recordings();
    }
}
```

### Get Recording Status (Web Interface)
```javascript
// Poll every 2 seconds during recording
async function updateRecordingStatus() {
    const response = await fetch('http://10.0.0.58:5000/api/v1/recordings/status');
    const status = await response.json();

    if (status.status === 'recording') {
        const elapsed = status.elapsed_seconds;
        const total = status.expected_duration;
        const percent = (elapsed / total) * 100;

        // Update progress bar
        document.getElementById('recordingProgress').style.width = percent + '%';
        document.getElementById('recordingTime').textContent =
            `${formatTime(elapsed)} / ${formatTime(total)}`;
    }
}
```

## Migration Guide

### For Existing Recordings

Recordings created before these changes won't have all metadata. The web interface should handle gracefully:

```python
# In annotation handler
duration = recording.get('duration_seconds') or 'Unknown'
playback_file = recording.get('playback_file') or 'N/A'
start_time = recording.get('start_time') or recording.get('timestamp')
```

### Database Updates

After rubix44 server is updated, run migration to backfill metadata:

```python
# scripts/migrate_rubix44_metadata.py
# Fetch updated history from rubix44
# Update recording_sessions table with new fields
```

## Contact & Questions

If implementing these changes, please coordinate with:
- Web interface maintainer (for integration testing)
- Database admin (for schema updates if needed)

## Status Tracking

- [ ] Stop recording endpoint implemented
- [ ] Status enhancements implemented
- [ ] History enhancements implemented
- [ ] Playback file metadata implemented
- [ ] Web interface updated to use new fields
- [ ] Integration testing complete
- [ ] Documentation updated

---

**Last Updated:** 2026-01-04
**Requested By:** Recording Management System v2.0
**Impact:** HIGH - Core functionality depends on these changes
