# Rubix44 File Access - Test Results

**Date**: January 17, 2026
**Server**: http://10.0.0.58:5000

---

## API Endpoint Test Results

### ✅ Working Endpoints

#### 1. `/api/v1/health`
**Status**: ✅ Working
```bash
curl http://10.0.0.58:5000/api/v1/health
```
**Response**:
```json
{
  "service": "Rubix Recorder API",
  "status": "healthy",
  "timestamp": "2026-01-17T10:27:00.000000"
}
```

#### 2. `/api/v1/recordings/history`
**Status**: ✅ Working Perfectly
**Result**: Returns 183 recordings with complete metadata
```bash
curl http://10.0.0.58:5000/api/v1/recordings/history
```

**Sample Response**:
```json
[
  {
    "id": "test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12",
    "prefix": "test_12hr_expexp_8ca89843_cycle9",
    "duration_seconds": 180.00,
    "start_time": "2026:01:14T22:57:12",
    "end_time": "2026:01:14T23:00:12",
    "files": [
      {
        "name": "test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_stereo.wav",
        "path": "recordings\\test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_stereo.wav",
        "size": 31752044,
        "modified": "2026-01-14T23:03:22.123456"
      },
      {
        "name": "test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_ch1.wav",
        "path": "recordings\\test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_ch1.wav",
        "size": 15876044,
        "modified": "2026-01-14T23:03:22.234567"
      },
      {
        "name": "test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_ch2.wav",
        "path": "recordings\\test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_ch2.wav",
        "size": 15876044,
        "modified": "2026-01-14T23:03:22.345678"
      }
    ],
    "sample_rate": 44100,
    "timestamp": "2026-01-14_22-57-12"
  }
]
```

**Key Observations**:
- Returns ALL recordings ever made (183 total)
- Includes complete file metadata (name, path, size, modified date)
- Uses Windows path format (`recordings\filename.wav`)
- Our experiment recordings are present with prefix starting with experiment ID

#### 3. `/api/v1/playback-files`
**Status**: ✅ Working
**Result**: Lists available playback audio files
```bash
curl http://10.0.0.58:5000/api/v1/playback-files
```

**Response**:
```json
[
  {
    "filename": "exp6 - 3 noise.wav",
    "path": "playback_files\\exp6 - 3 noise.wav",
    "size": 2770095456,
    "duration_seconds": 7213.79,
    "channels": 2,
    "sample_rate": 48000,
    "format": "WAV"
  }
]
```

### ❌ Not Working Endpoints

#### 1. `/api/v1/recordings/download/{filename}`
**Status**: ❌ Returns 404
**Tested**: `curl http://10.0.0.58:5000/api/v1/recordings/download/test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12_stereo.wav`
**Response**: `{"error":"File not found"}`

**Possible Reasons**:
1. Endpoint not fully implemented yet
2. Incorrect filename format expected
3. Files stored in location not accessible to download endpoint
4. Download feature planned but not released

#### 2. Direct Static File Serving
**Status**: ❌ Not enabled
**Tested**: `curl http://10.0.0.58:5000/recordings/filename.wav`
**Result**: 404

---

## Our Experiment Recordings Found

From the history endpoint, we found **72 recordings** from experiment exp_8ca89843:

**Sample IDs** (prefix pattern: `test_12hr_expexp_8ca89843_cycle{N}`):
- test_12hr_expexp_8ca89843_cycle9_2026-01-14_22-57-12
- test_12hr_expexp_8ca89843_cycle10_2026-01-14_23-03-12
- ... (70 more cycles)

**Files per recording**:
- `*_stereo.wav` - Combined stereo file (~31.7 MB for 180s at 44.1kHz)
- `*_ch1.wav` - Left channel only (~15.9 MB)
- `*_ch2.wav` - Right channel only (~15.9 MB)

---

## Recommended Solutions

### Solution 1: Use History API + Network File Share (IMMEDIATE)

Since the history API works perfectly and provides file metadata, we can:

1. **Query history to get file info**:
   ```python
   def get_recording_info(session_id: str):
       response = requests.get('http://10.0.0.58:5000/api/v1/recordings/history')
       recordings = response.json()
       for rec in recordings:
           if session_id in rec['id']:
               return rec
       return None
   ```

2. **Access files via network share**:
   ```bash
   # Mount Windows share (one-time setup)
   mkdir -p /Users/bernd/rubix44/recordings
   mount -t smbfs //10.0.0.58/recordings /Users/bernd/rubix44/recordings
   ```

3. **Convert Windows path to local mount path**:
   ```python
   def convert_path_for_local_access(windows_path: str) -> str:
       # Convert: recordings\file.wav -> /Users/bernd/rubix44/recordings/file.wav
       filename = windows_path.split('\\')[-1]
       return f"/Users/bernd/rubix44/recordings/{filename}"
   ```

### Solution 2: Request Download Endpoint Fix (MEDIUM-TERM)

The rubix44 server should implement the download endpoint properly:

**Server-side fix needed**:
```python
@app.route('/api/v1/recordings/download/<filename>')
def download_recording(filename):
    file_path = os.path.join(RECORDINGS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)
```

### Solution 3: Implement Custom File Transfer (ALTERNATIVE)

If download endpoint can't be fixed immediately:

1. **Add SCP/rsync support** on rubix44 server
2. **Use Python paramiko** to fetch files:
   ```python
   import paramiko

   def download_via_ssh(filename: str, local_path: str):
       ssh = paramiko.SSHClient()
       ssh.connect('10.0.0.58', username='user', password='pass')
       sftp = ssh.open_sftp()
       sftp.get(f'/path/to/recordings/{filename}', local_path)
       sftp.close()
       ssh.close()
   ```

---

## Immediate Action Items

### For Continuous Learning Pipeline

1. **✅ WORKING**: Query history API to verify recording exists
2. **❌ BLOCKED**: Direct file download (endpoint returns 404)
3. **⚠️ WORKAROUND NEEDED**: Access files via alternative method

### Recommended Next Steps

**Option A - Quick Fix** (If network share is available):
1. Mount rubix44 recordings directory via SMB/NFS
2. Update orchestrator to use mounted path
3. Continue with QC and feature extraction

**Option B - Proper Fix** (If download endpoint can be fixed):
1. Fix `/api/v1/recordings/download/{filename}` endpoint on rubix44 server
2. Test download with our recordings
3. Integrate download into orchestrator workflow

**Option C - SSH/SCP** (If server has SSH enabled):
1. Enable SSH on rubix44 server (10.0.0.58)
2. Use paramiko/scp to transfer files
3. Integrate into orchestrator

---

## Code Integration

Once file access is resolved, integrate like this:

```python
async def process_recording(self, session_id: str, cycle_number: int):
    # Step 1: Get recording info from history
    history_response = requests.get(f"{self.rubix_url}/api/v1/recordings/history")
    recordings = history_response.json()

    recording_info = None
    for rec in recordings:
        if session_id in rec['id']:
            recording_info = rec
            break

    if not recording_info:
        self.logger.error(f"Recording {session_id} not found in history")
        return False

    # Step 2: Get stereo file info
    stereo_file = None
    for file_info in recording_info.get('files', []):
        if '_stereo.wav' in file_info['name']:
            stereo_file = file_info
            break

    if not stereo_file:
        self.logger.error(f"No stereo file found for {session_id}")
        return False

    # Step 3: Download or access file
    # OPTION A: Network mount
    local_path = f"/Users/bernd/rubix44/recordings/{stereo_file['name']}"

    # OPTION B: Download (when endpoint works)
    # local_path = await self.download_recording(stereo_file['name'])

    # OPTION C: SSH transfer
    # local_path = await self.scp_download(stereo_file['name'])

    # Step 4: Verify file exists
    if not Path(local_path).exists():
        self.logger.error(f"File not accessible: {local_path}")
        return False

    # Step 5: Proceed with QC
    self.logger.info(f"Processing {local_path} ({stereo_file['size']} bytes)")
    qc_result = await self.qc_validator.validate_recording(
        session_id=session_id,
        wav_path=Path(local_path),
        duration_seconds=recording_info['duration_seconds'],
        channel_1_class=self.config['channel_1_expected_class'],
        channel_2_class=self.config['channel_2_expected_class']
    )

    return qc_result.passed
```

---

## Summary

**What Works**: ✅
- Recording creation
- Progress tracking
- History query with complete metadata
- Playback file listing

**What Doesn't Work**: ❌
- File download endpoint
- Direct static file access

**What's Needed**: ⚠️
- Network file share OR
- Fixed download endpoint OR
- SSH/SCP file transfer

**Best Immediate Solution**: Setup network mount of rubix44 recordings directory, then update orchestrator to use mounted paths.
