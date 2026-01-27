# Rubix44 API Integration Review

**Date**: 2026-01-25 (Updated)
**Original Date**: 2026-01-11
**API Version**: v1.1.0
**Our Implementation**: src/continuous/rubix44_data_provider.py

## Executive Summary

The Rubix44 API provides **comprehensive metadata** for recording management. This document reviews what's available and how our implementation uses it.

## Rubix44 API - Available Fields

### Status Endpoint (`GET /api/v1/status`)

**During Recording:**

| Field | Available | Location | Notes |
| ----- | --------- | -------- | ----- |
| `duration` (requested) | ✅ Yes | `recording.duration` | Duration passed in start request |
| `elapsed_seconds` | ✅ Yes | `recording.elapsed_seconds` | Calculated in real-time |
| `progress_percent` | ✅ Yes | `recording.progress_percent` | Percentage completion |
| `expected_duration` | ✅ Yes | `recording.expected_duration` | Same as duration |
| `input_device` | ✅ Yes | `recording.input_device` | Device ID (integer) |
| `output_device` | ✅ Yes | `recording.output_device` | Device ID (integer) |
| Full device info | ✅ Yes | `rubix.input_device`, `rubix.output_device` | Name, channels, sample_rate |
| `expected_end_time` | ❌ No | - | Calculate: `start_time + duration` |
| `auto_stop_enabled` | ❌ No | - | Always enabled via watchdog |

**Note:** Auto-stop is always active via the watchdog mechanism.

### History Endpoint (`GET /api/v1/recordings/history`)

All v1.1.0 metadata fields are available:

| Field | Available | Notes |
| ----- | --------- | ----- |
| `start_time` | ✅ Yes | ISO format timestamp |
| `end_time` | ✅ Yes | ISO format timestamp |
| `duration_seconds` | ✅ Yes | Actual recording duration |
| `playback_file` | ✅ Yes | Which stimulus was used |
| `sample_rate` | ✅ Yes | Audio sample rate |
| `files[].modified` | ✅ Yes | File modification timestamp |

## Our Implementation Coverage

### ✅ Currently Implemented

| Endpoint | Status | Implementation |
| -------- | ------ | -------------- |
| `GET /api/v1/status` | ✅ Used | Health checks, status polling |
| `GET /api/v1/config` | ✅ Used | Server configuration retrieval |
| `GET /api/v1/recordings/history` | ✅ Used | Core polling mechanism |
| `GET /api/v1/recordings/{filename}` | ✅ Used | File download with streaming |
| `POST /api/v1/recordings/start` | ✅ Used | Remote recording control |
| `POST /api/v1/recordings/stop` | ✅ Used | Stop recordings remotely |

### ⚠️ Could Use More Metadata

| Endpoint | Status | Opportunity |
| -------- | ------ | ----------- |
| `GET /api/v1/recordings/history` | ⚠️ Partial | Could extract more v1.1.0 metadata |
| `GET /api/v1/status` | ⚠️ Partial | Could log elapsed_seconds, progress_percent |

### ❌ Not Implemented (Optional)

| Endpoint | Status | Impact |
| -------- | ------ | ------ |
| `GET /api/v1/health` | ❌ Optional | Using /status works fine |
| `GET /api/v1/devices` | ❌ Optional | For advanced diagnostics |
| `GET /api/v1/devices/rubix` | ❌ Optional | For device validation |
| `GET /api/v1/playback-files` | ❌ Optional | For playback file validation |
| `PUT /api/v1/config` | ❌ Optional | For remote configuration |

## Detailed Analysis

### 1. Health Check Endpoint (❌ Critical Issue)

**Current Implementation:**
```python
# rubix44_data_provider.py:44-56
def health_check(self) -> bool:
    try:
        response = requests.get(f"{self.api_base}/recordings/status", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

**Issue:** Using `/recordings/status` instead of dedicated `/health` endpoint.

**API Documentation:**
```
GET /api/v1/health
Response: {
  "status": "healthy",
  "timestamp": "2026-01-03T12:23:41.584Z",
  "service": "Rubix Recorder API"
}
```

**Recommendation:**
```python
def health_check(self) -> bool:
    """Check if API server is healthy using dedicated health endpoint."""
    try:
        response = requests.get(f"{self.api_base}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.debug(f"Health check passed: {data}")
            return data.get('status') == 'healthy'
        return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

### 2. Recording History - Missing Enhanced Metadata (⚠️ Important)

**Current Implementation:**
```python
# rubix44_data_provider.py:80-93
def get_recording_history(self) -> List[Dict]:
    response = requests.get(f"{self.api_base}/recordings/history", timeout=10)
    response.raise_for_status()
    return response.json()
```

**API v1.1.0 Enhanced Response:**
```json
[
  {
    "id": "recording_2026-01-04_16-30-00",
    "prefix": "recording",
    "timestamp": "2026-01-04_16-30-00",
    "start_time": "2026-01-04T16:30:00",        // NEW
    "end_time": "2026-01-04T17:30:25",          // NEW
    "duration_seconds": 3625.5,                  // NEW
    "playback_file": "noise_baseline.wav",       // NEW
    "sample_rate": 44100,                        // NEW
    "files": [
      {
        "name": "recording_stereo.wav",
        "path": "/full/path/to/file",
        "size": 123456789,
        "modified": "2026-01-04T17:30:25"        // NEW
      }
    ]
  }
]
```

**What We're Missing:**
- `start_time` and `end_time` (ISO format timestamps)
- `duration_seconds` (actual recording duration)
- `playback_file` (which stimulus was used)
- `sample_rate` (for validation)
- `modified` timestamp for files

**Impact:**
- Cannot validate recording duration
- Cannot track which playback file was used (important for experiment tracking)
- Cannot detect if files were modified after recording
- Missing valuable QC metadata

**Recommendation:**
```python
def _process_session(self, session: Dict) -> bool:
    """Process a single recording session with enhanced metadata."""
    session_id = session['id']

    # Extract v1.1.0 metadata
    duration_sec = session.get('duration_seconds', 0)
    playback_file = session.get('playback_file', 'unknown')
    sample_rate = session.get('sample_rate', 44100)
    start_time = session.get('start_time')
    end_time = session.get('end_time')

    logger.info(f"Processing session: {session_id}")
    logger.info(f"  Duration: {duration_sec:.1f} sec")
    logger.info(f"  Playback: {playback_file}")
    logger.info(f"  Sample rate: {sample_rate} Hz")

    # Validate duration (e.g., reject very short recordings)
    if duration_sec < 60:
        logger.warning(f"Recording too short ({duration_sec}s), skipping")
        return False

    # Store playback file in metadata for experiment tracking
    metadata = self._get_metadata(session_id)
    if metadata:
        # Could update MariaDB with playback_file info
        self._update_playback_metadata(session_id, playback_file)

    # ... rest of processing
```

### 3. Device Management (❌ Not Implemented)

**Missing Endpoints:**
- `GET /api/v1/devices` - List all audio devices
- `GET /api/v1/devices/rubix` - Verify Rubix44 is connected

**Use Cases:**
1. **Startup Validation**: Verify Rubix44 is available before starting monitoring
2. **Health Monitoring**: Detect if Rubix44 gets disconnected during operation
3. **Error Diagnostics**: Better error messages when device issues occur

**Recommendation - Add to startup checks:**
```python
class Rubix44Client:
    def get_rubix_device(self) -> Optional[Dict]:
        """Get Rubix44 device information."""
        try:
            response = requests.get(f"{self.api_base}/devices/rubix", timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get('found'):
                logger.info(f"Rubix44 found: Input device {data['input_device']}")
                logger.info(f"  Channels: {data['input_device_info']['channels']}")
                logger.info(f"  Sample rate: {data['input_device_info']['sample_rate']}")
                return data
            else:
                logger.error("Rubix44 not found!")
                return None
        except Exception as e:
            logger.error(f"Error checking Rubix device: {e}")
            return None

# In Rubix44DataProvider.__init__:
def __init__(self, ...):
    # ... existing init code ...

    # Verify Rubix44 is available
    device_info = self.client.get_rubix_device()
    if not device_info:
        logger.warning("Rubix44 device not detected at startup!")
    else:
        self.device_info = device_info
```

### 4. Playback Files Listing (❌ Not Implemented)

**Missing Endpoint:**
- `GET /api/v1/playback-files`

**API Response:**
```json
[
  {
    "filename": "noise_baseline.wav",
    "path": "playback_files/noise_baseline.wav",
    "size": 987654321,
    "duration_seconds": 120.5,
    "sample_rate": 44100,
    "channels": 2,
    "format": "WAV",
    "modified": "2026-01-01T12:00:00.000Z"
  }
]
```

**Use Cases:**
1. **Validation**: Check if required playback files exist before starting recordings
2. **Experiment Planning**: Query available stimuli
3. **Health Checks**: Detect missing or corrupted playback files

**Recommendation:**
```python
class Rubix44Client:
    def get_playback_files(self) -> List[Dict]:
        """Get list of available playback files with metadata."""
        response = requests.get(f"{self.api_base}/playback-files", timeout=5)
        response.raise_for_status()
        return response.json()

    def validate_playback_file(self, filename: str) -> bool:
        """Check if a specific playback file exists and is valid."""
        files = self.get_playback_files()
        for f in files:
            if f['filename'] == filename:
                logger.info(f"Playback file validated: {filename}")
                logger.info(f"  Duration: {f['duration_seconds']:.1f}s")
                logger.info(f"  Sample rate: {f['sample_rate']} Hz")
                return True
        logger.error(f"Playback file not found: {filename}")
        return False
```

### 5. Remote Recording Control (❌ Not Implemented)

**Missing Endpoints:**
- `POST /api/v1/recordings/start`
- `POST /api/v1/recordings/stop`

**Current Status:** We only poll for completed recordings. Cannot trigger recordings.

**Use Cases:**
1. **Scheduled Recording**: Start recordings at specific times
2. **Adaptive Experiments**: Trigger recordings based on model performance
3. **Remote Operation**: Control recorder from continuous learning pipeline
4. **Emergency Stop**: Stop problematic recordings

**Recommendation - Future Enhancement:**
```python
class Rubix44Client:
    def start_recording(self,
                       playback_file: str,
                       duration: int = 3600,
                       output_prefix: str = "xor_recording",
                       sample_rate: int = 44100) -> Dict:
        """
        Start a new recording session.

        Args:
            playback_file: Path to playback file (relative to server's playback_files/)
            duration: Recording duration in seconds (default: 1 hour)
            output_prefix: Prefix for output files
            sample_rate: Sample rate in Hz

        Returns:
            Session info dictionary
        """
        payload = {
            "playback_file": playback_file,
            "duration": duration,
            "output_prefix": output_prefix,
            "sample_rate": sample_rate
        }

        response = requests.post(
            f"{self.api_base}/recordings/start",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

    def stop_recording(self) -> Dict:
        """
        Stop the current recording session.

        Returns:
            Session completion info with files and duration
        """
        response = requests.post(
            f"{self.api_base}/recordings/stop",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
```

**Configuration Addition:**
```yaml
orchestration:
  rubix44:
    # ... existing config ...

    # Remote recording control (optional)
    remote_control_enabled: false  # Enable remote recording triggers
    scheduled_recordings:  # Optional scheduled recording times
      - time: "02:00"
        duration: 3600
        playback_file: "playback_files/noise_baseline.wav"
        output_prefix: "nightly_recording"
```

### 6. Recording Status - Enhanced Information (✅ Available)

**API Response (when recording via `GET /api/v1/status`):**

```json
{
  "recording": {
    "status": "recording",
    "session_id": "recording_2026-01-04_16-30-00",
    "start_time": "2026-01-04T16:30:00",
    "duration": 240,                           // ✅ Requested duration
    "elapsed_seconds": 125.5,                  // ✅ Available
    "expected_duration": 240,                  // ✅ Available (same as duration)
    "progress_percent": 52.3,                  // ✅ Available
    "playback_file": "noise_baseline.wav",     // ✅ Available
    "output_prefix": "recording",
    "sample_rate": 44100,
    "input_device": 1,                         // ✅ Device ID
    "output_device": 5                         // ✅ Device ID
  },
  "rubix": {
    "connected": true,
    "input_device": {"id": 1, "name": "Line (Roland Rubix 44)", ...},
    "output_device": {"id": 5, "name": "Speakers (Roland Rubix 44)", ...}
  }
}
```

**All key fields are available:**

- ✅ `elapsed_seconds` - Current progress (real-time)
- ✅ `duration` / `expected_duration` - Requested duration
- ✅ `progress_percent` - Percentage completion
- ✅ `input_device` / `output_device` - Device IDs during recording
- ✅ Full device details in `rubix` section

**Note:** `expected_end_time` is not returned but can be calculated as `start_time + duration`.

**Example monitoring code:**

```python
def monitor_active_recording(self) -> Optional[Dict]:
    """Monitor active recording with progress information."""
    status = self.get_status()  # GET /api/v1/status
    recording = status.get('recording', {})

    if recording.get('status') != 'recording':
        return None

    return {
        'session_id': recording.get('session_id'),
        'elapsed_seconds': recording.get('elapsed_seconds', 0),
        'expected_duration': recording.get('duration', 0),
        'progress_percent': recording.get('progress_percent', 0),
        'playback_file': recording.get('playback_file'),
        'remaining_seconds': recording.get('duration', 0) - recording.get('elapsed_seconds', 0)
    }
```

## Priority Recommendations

### High Priority - ✅ DONE

1. **Recording Status Fields** ✅
   - `elapsed_seconds`, `duration`, `progress_percent` all available
   - Device info available in `rubix` section

2. **History Metadata** ✅
   - `duration_seconds`, `playback_file`, `sample_rate` all available
   - `start_time`, `end_time` timestamps available

3. **Remote Recording Control** ✅
   - `POST /api/v1/recordings/start` implemented
   - `POST /api/v1/recordings/stop` implemented

### Medium Priority - Optional Enhancements

4. **Better use of available metadata**
   - Our code could extract more of the available v1.1.0 fields
   - Log `playback_file` for experiment tracking
   - Use `progress_percent` for monitoring

5. **Device Validation on Startup**
   - Could use `/devices/rubix` to verify device connectivity
   - Not critical - current health checks work

### Low Priority - Nice to Have

6. **Playback Files Validation**
   - Use `/playback-files` to validate before starting
   - Prevents configuration errors

7. **Configuration Management**
   - Use `PUT /config` for remote configuration
   - Rarely needed in practice

## Testing Checklist

After implementing changes, verify:

- [ ] Health check uses correct `/health` endpoint
- [ ] Health check validates `status == 'healthy'`
- [ ] Recording history extracts all v1.1.0 fields
- [ ] Duration validation works (reject short recordings)
- [ ] Playback file is logged for each session
- [ ] Device check runs on startup
- [ ] Error messages improved with device info
- [ ] Playback file validation available
- [ ] Unit tests updated for new client methods
- [ ] Integration test with real API passes

## Updated Test Script

Enhance `scripts/test_rubix44_provider.py` to test new features:

```python
def test_api_connection(api_url: str):
    """Test connection to rubix44-recorder API with v1.1.0 features"""
    client = Rubix44Client(api_url)

    # 1. Health check (using correct endpoint)
    logger.info("1. Health Check (/health)...")
    healthy = client.health_check()
    logger.info(f"   API Healthy: {healthy}")

    # 2. Device validation
    logger.info("2. Device Check (/devices/rubix)...")
    device = client.get_rubix_device()
    if device:
        logger.info(f"   Found Rubix44: {device['input_device_info']['name']}")

    # 3. Playback files
    logger.info("3. Playback Files (/playback-files)...")
    files = client.get_playback_files()
    logger.info(f"   Available files: {len(files)}")
    for f in files:
        logger.info(f"     - {f['filename']}: {f['duration_seconds']:.1f}s")

    # 4. Recording history with enhanced metadata
    logger.info("4. Recording History (enhanced)...")
    history = client.get_recording_history()
    for session in history[:3]:
        logger.info(f"   Session: {session['id']}")
        logger.info(f"     Duration: {session.get('duration_seconds', 0):.1f}s")
        logger.info(f"     Playback: {session.get('playback_file', 'unknown')}")
        logger.info(f"     Sample rate: {session.get('sample_rate', 0)} Hz")

    # 5. Active recording status
    logger.info("5. Recording Status (enhanced)...")
    status = client.get_recording_status()
    if status.get('status') == 'recording':
        elapsed = status.get('elapsed_seconds', 0)
        expected = status.get('expected_duration', 0)
        progress = (elapsed / expected * 100) if expected > 0 else 0
        logger.info(f"   Recording in progress: {progress:.1f}% complete")
    else:
        logger.info(f"   Status: {status.get('status')}")
```

## Configuration Updates

Add to `config/continuous_learning_config.yaml`:

```yaml
orchestration:
  rubix44:
    api_url: http://10.0.0.58:5000
    poll_interval_minutes: 5
    download_dir: data/continuous/recordings
    output_prefix_filter: null
    cleanup_after_processing: false

    # NEW: Enhanced validation
    validate_device_on_startup: true  # Check Rubix44 is connected
    validate_playback_files: true     # Check required playback files exist
    min_recording_duration_sec: 60    # Reject recordings shorter than 1 minute

    # NEW: Monitoring
    monitor_active_recordings: true   # Log progress of active recordings
    monitor_interval_minutes: 15      # Check active recording progress every 15 min
```

## Backward Compatibility

All recommended changes are backward compatible:

- New fields from API are optional (use `.get()` with defaults)
- Old tests continue to work with mock responses
- Can deploy incrementally (health check fix first, then enhancements)
- No breaking changes to existing interfaces

## Summary

The Rubix44 API **provides all the metadata we need**. Key findings:

### What's Available (Server-Side)

| Category | Status | Notes |
| -------- | ------ | ----- |
| Recording duration (requested) | ✅ Available | `recording.duration` |
| Elapsed time (real-time) | ✅ Available | `recording.elapsed_seconds` |
| Progress percentage | ✅ Available | `recording.progress_percent` |
| Device info during recording | ✅ Available | ID in recording, full details in rubix section |
| History metadata | ✅ Available | duration_seconds, playback_file, timestamps |
| Auto-stop mechanism | ✅ Active | Watchdog always running |
| Expected end time | ❌ Not returned | Calculate: `start_time + duration` |

### Client-Side Calculations

Two fields are not returned but can be easily calculated:

1. **`expected_end_time`**: `start_time + duration`
2. **`auto_stop_enabled`**: Always `true` (watchdog is always active)

### Recommendation

The API provides sufficient information for debugging recording issues. If a recording doesn't stop at the expected time, the issue is likely:

1. **Watchdog failure** on the server side
2. **Network issues** preventing status updates
3. **Server crash/hang** during recording

The client should use `elapsed_seconds` > `duration` as a timeout trigger and call `POST /recordings/stop` if the server hasn't auto-stopped.
