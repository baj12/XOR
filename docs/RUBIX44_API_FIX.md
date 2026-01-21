# Rubix44 API Integration Fix

## Problem Summary

Experiment `exp_784e2ad3` failed on **all 33 cycles** with the error:
```
ERROR - Failed to start recording: No session_id returned from rubix44
```

## Root Cause

The continuous learning orchestrator was expecting the rubix44 API to return the session ID in a different format than what the API actually returns.

### Expected (Code):
```json
{
  "session_id": "20260114_175618"
}
```

### Actual (API):
```json
{
  "message": "Recording started",
  "session": {
    "id": "20260114_175618",
    "status": "recording",
    "duration": 180,
    "playback_file": "exp6 - 3 noise.wav",
    "output_prefix": "test_cycle1",
    ...
  }
}
```

**The session ID is nested**: `response['session']['id']` not `response['session_id']`

## The Fix

**File**: [src/continuous/recording_orchestrator.py](../src/continuous/recording_orchestrator.py#L191-L204)

**Before** (lines 191-195):
```python
result = response.json()
session_id = result.get('session_id')

if not session_id:
    raise ValueError("No session_id returned from rubix44")
```

**After** (lines 191-204):
```python
result = response.json()

# API returns session info nested under 'session' key
# The session ID is in session['id']
session_data = result.get('session', {})
session_id = session_data.get('id')

# Fallback: try direct session_id key for backward compatibility
if not session_id:
    session_id = result.get('session_id')

if not session_id:
    self.logger.error(f"API response: {result}")
    raise ValueError("No session_id returned from rubix44")
```

## Testing

### Verified API Response

```bash
curl -X POST 'http://10.0.0.58:5000/api/v1/recordings/start' \
  -H 'Content-Type: application/json' \
  -d '{"playback_file": "exp6 - 3 noise.wav", "duration": 180, "output_prefix": "test"}'
```

**Response** (Status 202):
```json
{
  "message": "Recording started",
  "session": {
    "id": "20260114_175618",
    "human_id": "fine-raven-6828",
    "status": "recording",
    "channels": 2,
    "sample_rate": 44100,
    "duration": 180,
    "playback_file": "exp6 - 3 noise.wav",
    "output_prefix": "test",
    "start_time": "2026-01-14T17:56:18.763946",
    "progress_percent": 0.0,
    "elapsed_seconds": 0.0
  }
}
```

### Health Check
```bash
curl http://10.0.0.58:5000/api/v1/health
```
✅ Server is healthy and responding correctly

## GitHub Repository

**Repository**: https://github.com/baj12/rubix44-recorder
**Latest Commits**: January 11, 2026
- Added recording deletion and storage transfer endpoints
- Implemented human-readable session IDs (e.g., "swift-panda-2347")
- Enhanced status reporting with complete session metadata

The rubix44-recorder API structure is current and follows the documented format. The issue was in our integration code, not the server.

**Server Location**: http://10.0.0.58:5000

**API Endpoints Used**:
- `POST /api/v1/recordings/start` - Start a new recording
- `GET /api/v1/recordings/status` - Check recording status
- `GET /api/v1/health` - Health check

**Official Documentation**:
- [API Docs](https://github.com/baj12/rubix44-recorder/blob/main/API_DOCS.md)
- [README](https://github.com/baj12/rubix44-recorder/blob/main/README.md)

## What Changed in the API

According to the [GitHub repository](https://github.com/baj12/rubix44-recorder), recent updates (January 2026) enhanced the API:

1. **Richer session object** with more metadata
2. **Human-readable ID** (`human_id`: "swift-panda-2347" format) - Added January 11, 2026
3. **Timestamp tracking** (`start_time`, `end_time`)
4. **Progress tracking** (`progress_percent`, `elapsed_seconds`) - Added January 4, 2026
5. **Status information** (`status`, `error`)
6. **File metadata** (`files` array with file details)
7. **Device information** (`input_device`, `output_device`)

The API follows the documented format from [API_DOCS.md](https://github.com/baj12/rubix44-recorder/blob/main/API_DOCS.md)

## Improvements Made

### 1. Better Error Logging
Now logs the full API response when session_id is missing, making debugging easier.

### 2. Backward Compatibility
Maintains fallback to `result.get('session_id')` in case the API changes back or for testing with mock servers.

### 3. Nested Data Handling
Properly extracts the session ID from the nested structure.

## Impact on Existing Experiments

### Before Fix
- ✗ All cycles failed immediately
- ✗ No recordings were ever started
- ✗ 0 samples collected
- ✗ Experiment completed without any work done

### After Fix
- ✓ Recording will start successfully
- ✓ Session ID will be correctly extracted
- ✓ Cycles will proceed through full pipeline:
  1. Start recording
  2. Wait for completion
  3. Run Auto-QC
  4. Extract features
  5. Train model

## Testing the Fix

To test with a new experiment:

```bash
# Start web server
python web/app.py --port 5001 --host 0.0.0.0

# Create experiment via web UI at http://localhost:5001/continuous
# - Duration: 0.01 weeks (~16 cycles, 96 minutes)
# - Interval: 6 minutes
# - Playback file: "exp6 - 3 noise.wav"
```

Monitor logs:
```bash
tail -f logs/continuous/<experiment_id>.log
```

Expected log output:
```
INFO - Starting cycle 1
INFO - Recording started: 20260114_180500
INFO - Waiting for recording 20260114_180500 to complete...
INFO - Recording 20260114_180500 completed
INFO - Running Auto-QC on /Users/bernd/rubix44/recordings/20260114_180500_stereo.wav
```

## Related Files

- **Fix applied**: [src/continuous/recording_orchestrator.py](../src/continuous/recording_orchestrator.py)
- **Failed experiment**: exp_784e2ad3 (0/33 cycles succeeded)
- **Log file**: logs/continuous/exp_784e2ad3.log

## Next Steps

1. ✅ **Fix applied** - Code updated to handle nested session response
2. ⏭️ **Test with new experiment** - Create a test experiment to verify the fix
3. ⏭️ **Document rubix44-recorder API** - Consider adding API documentation if not already available
4. ⏭️ **Add integration tests** - Create tests for the API response parsing

## API Response Schema

For future reference, here's the expected response structure:

```typescript
// POST /api/v1/recordings/start
interface StartRecordingResponse {
  message: string;
  session: {
    id: string;                    // Session ID (timestamp format: YYYYMMDD_HHMMSS)
    human_id: string;              // Human-readable ID (e.g., "fine-raven-6828")
    status: "recording" | "idle" | "error";
    channels: number;              // Number of audio channels
    sample_rate: number;           // Sample rate in Hz
    duration: number;              // Expected duration in seconds
    playback_file: string;         // Name of playback file
    output_prefix: string;         // Output file prefix
    start_time: string;            // ISO 8601 timestamp
    end_time: string | null;       // ISO 8601 timestamp when complete
    progress_percent: number;      // Progress percentage (0-100)
    elapsed_seconds: number;       // Elapsed time in seconds
    expected_duration: number;     // Expected total duration
    error: string | null;          // Error message if failed
    files: string[];               // List of generated files
    input_device: string | null;   // Input device name
    output_device: string | null;  // Output device name
  };
}
```

## Conclusion

The issue was a **mismatch between the expected and actual API response structure**. The fix properly extracts the session ID from the nested `session` object while maintaining backward compatibility. All future experiments should now successfully start recordings and proceed through the full continuous learning pipeline.
