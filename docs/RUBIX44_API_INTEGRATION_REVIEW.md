# Rubix44 API Integration Review

**Date**: 2026-01-11
**API Version**: v1.1.0
**Our Implementation**: src/continuous/rubix44_data_provider.py

## Executive Summary

Our current implementation uses **most** of the Rubix44 API features but is **missing several v1.1.0 enhancements** that provide valuable metadata. This review identifies gaps and recommends improvements.

## API Feature Coverage

### ✅ Currently Implemented

| Endpoint | Status | Implementation |
|----------|--------|----------------|
| `GET /api/v1/recordings/status` | ✅ Used | Health checks, status polling |
| `GET /api/v1/config` | ✅ Used | Server configuration retrieval |
| `GET /api/v1/recordings/history` | ✅ Used | Core polling mechanism |
| `GET /api/v1/recordings/{filename}` | ✅ Used | File download with streaming |

### ⚠️ Partially Implemented

| Endpoint | Status | Issue |
|----------|--------|-------|
| `GET /api/v1/recordings/history` | ⚠️ Partial | Not using v1.1.0 enhanced metadata |
| `GET /api/v1/recordings/status` | ⚠️ Partial | Not capturing elapsed_seconds, expected_duration |

### ❌ Not Implemented

| Endpoint | Status | Impact |
|----------|--------|--------|
| `GET /api/v1/health` | ❌ Missing | Using wrong endpoint for health checks |
| `GET /api/v1/devices` | ❌ Missing | Cannot verify Rubix44 connectivity |
| `GET /api/v1/devices/rubix` | ❌ Missing | Cannot confirm correct device in use |
| `GET /api/v1/playback-files` | ❌ Missing | Cannot validate playback file availability |
| `PUT /api/v1/config` | ❌ Missing | Cannot remotely configure recorder |
| `POST /api/v1/recordings/start` | ❌ Missing | Cannot trigger recordings remotely |
| `POST /api/v1/recordings/stop` | ❌ Missing | Cannot stop recordings remotely |

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

### 6. Recording Status - Enhanced Information (⚠️ Minor)

**Current Implementation:**
```python
# rubix44_data_provider.py:69-78
def get_recording_status(self) -> Dict:
    response = requests.get(f"{self.api_base}/recordings/status", timeout=5)
    response.raise_for_status()
    return response.json()
```

**API v1.1.0 Response (when recording):**
```json
{
  "status": "recording",
  "session_id": "recording_2026-01-04_16-30-00",
  "start_time": "2026-01-04T16:30:00",
  "elapsed_seconds": 125.5,                    // NEW
  "playback_file": "noise_baseline.wav",       // NEW
  "output_prefix": "recording",
  "expected_duration": 3600,                   // NEW
  "sample_rate": 44100
}
```

**What We're Missing:**
- `elapsed_seconds` - Current progress
- `expected_duration` - Total expected duration
- Progress percentage calculation

**Use Case:** Better monitoring when recordings are in progress.

**Recommendation:**
```python
def monitor_active_recording(self) -> Optional[Dict]:
    """
    Monitor active recording with progress information.

    Returns:
        Progress dict with elapsed time and percentage, or None if idle
    """
    status = self.get_recording_status()

    if status.get('status') != 'recording':
        return None

    elapsed = status.get('elapsed_seconds', 0)
    expected = status.get('expected_duration', 0)
    progress_pct = (elapsed / expected * 100) if expected > 0 else 0

    return {
        'session_id': status.get('session_id'),
        'elapsed_seconds': elapsed,
        'expected_duration': expected,
        'progress_percent': progress_pct,
        'playback_file': status.get('playback_file'),
        'remaining_seconds': expected - elapsed
    }
```

## Priority Recommendations

### High Priority (Implement Now)

1. **Fix Health Check Endpoint**
   - Change from `/recordings/status` to `/health`
   - Validate response structure
   - **Effort:** 15 minutes
   - **Impact:** Correct API usage

2. **Extract Enhanced History Metadata**
   - Use `duration_seconds`, `playback_file`, `sample_rate`
   - Add validation based on duration
   - Log playback file for experiment tracking
   - **Effort:** 1 hour
   - **Impact:** Better QC and experiment tracking

3. **Add Device Validation**
   - Implement `get_rubix_device()` in client
   - Check device on startup
   - **Effort:** 30 minutes
   - **Impact:** Better error detection and diagnostics

### Medium Priority (Next Sprint)

4. **Playback Files Validation**
   - Implement `get_playback_files()`
   - Add startup validation
   - **Effort:** 30 minutes
   - **Impact:** Prevent recording failures

5. **Enhanced Status Monitoring**
   - Use `elapsed_seconds` and `expected_duration`
   - Add progress logging for active recordings
   - **Effort:** 45 minutes
   - **Impact:** Better visibility during recordings

### Low Priority (Future Enhancement)

6. **Remote Recording Control**
   - Implement `start_recording()` and `stop_recording()`
   - Add scheduled recording support
   - **Effort:** 2-3 hours
   - **Impact:** Enables adaptive experiments

7. **Configuration Management**
   - Implement `PUT /config` endpoint
   - Allow remote recorder configuration
   - **Effort:** 1 hour
   - **Impact:** More flexible deployment

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

Our current implementation covers the **core functionality** (polling and downloading) but misses several **v1.1.0 enhancements** that provide valuable metadata and validation capabilities. The recommended changes are straightforward and provide significant improvements in:

1. **Correctness** - Using proper health check endpoint
2. **Reliability** - Device validation and playback file checks
3. **Observability** - Duration, playback file, and progress tracking
4. **Experiment Tracking** - Recording which stimulus was used
5. **Quality Control** - Duration-based filtering and validation

**Estimated Total Effort:** 3-4 hours for high and medium priority items.

**Recommended Approach:**
1. Fix health check (15 min) ← Do this now
2. Add device validation (30 min) ← Do this now
3. Extract enhanced metadata (1 hour) ← Do this now
4. Remaining enhancements ← Next sprint
