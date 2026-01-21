# End-to-End Pipeline Test with File Download

**Date**: January 17, 2026
**Test Experiment**: exp_ee578709
**Purpose**: Validate complete continuous learning pipeline with file download functionality

---

## Test Objectives

This test validates all bug fixes from the 12-hour experiment (exp_8ca89843) are working correctly:

1. ✅ **Beaker role ENUM fix**: Changed default from 'none' to 'not_used'
2. ✅ **StereoChannelProcessor config fix**: Added config object creation
3. 🔄 **File download implementation**: NEW - Download WAV files from rubix44 server
4. ✅ **Timeout fix**: Increased from 10s to 30s
5. ✅ **API response parsing**: Handles nested session format

---

## Test Configuration

```yaml
Experiment ID: exp_ee578709
Name: Pipeline Test with Download
Target Duration: 0.007 weeks (~1 hour)
Recording Interval: 6 minutes
Recording Duration: 180 seconds (3 minutes)
Total Cycles Expected: 11

Channel Configuration:
  Channel 1: lavender (class 1)
  Channel 2: empty (class 0)

Rubix44 Configuration:
  Server: http://10.0.0.58:5000
  Playback File: exp6 - 3 noise.wav
  Download Directory: /Users/bernd/rubix44/recordings
```

---

## Pipeline Stages

Each cycle will go through these stages:

### Stage 1: Start Recording (via Rubix44 API)
```
POST /api/v1/recordings/start
{
  "playback_file": "exp6 - 3 noise.wav",
  "duration": 180,
  "output_prefix": "continuous_expexp_ee578709_cycle1"
}
```

**Expected**:
- Status 202 (Accepted)
- Response contains `session.id` and `session.human_id`
- Orchestrator logs session ID

### Stage 2: Poll Recording Progress
```
GET /api/v1/recordings/status
```

**Expected**:
- Status reports: recording → 16.7% → 33.4% → 50.1% → 66.8% → 83.4% → 100% → completed
- Progress logged every 30 seconds
- Human-readable ID logged

### Stage 3: Create Metadata in Database
```
INSERT INTO recording_sessions (session_id, recording_date, ...)
```

**Expected**:
- recording_sessions entry created
- recording_cycles entry updated with session_id (after parent exists - FK fix)
- Beaker roles set to 'not_used' (ENUM fix)
- Channel substances mapped to class labels (lavender=1, empty=0)

### Stage 4: Download Recording File (NEW!)
```
GET /api/v1/recordings/{session_id}_stereo.wav
```

**Expected**:
- File downloaded from rubix44 server to local directory
- File size ~30MB for 180s at 44.1kHz stereo
- Orchestrator logs download success with file size
- File saved to: `/Users/bernd/rubix44/recordings/{session_id}_stereo.wav`

### Stage 5: Quality Control Validation
```python
qc_validator.validate_recording(
    session_id=session_id,
    wav_path=local_wav_path,
    duration_seconds=180,
    channel_1_class=1,  # lavender
    channel_2_class=0   # empty
)
```

**Expected**:
- Processor instantiated with config (config fix)
- WAV file read from downloaded location
- Stereo channels separated
- Features extracted from each channel
- UMAP visualization generated
- Class separation score calculated
- QC result: PASS or FAIL with detailed metrics

### Stage 6: Feature Extraction (if QC passes)
```python
processor.process_stereo_recording(
    wav_path=local_wav_path,
    channel_1_class=1,
    channel_2_class=0
)
```

**Expected**:
- Features extracted for both channels
- Features stored in database
- Feature counts logged

---

## Success Criteria

The test is successful if:

1. **Recording Start**: ✅ All 11 cycles start successfully
2. **Progress Tracking**: ✅ Progress updates logged during recording
3. **Metadata Creation**: ✅ Database entries created for all cycles
4. **File Download**: ✅ All 11 WAV files downloaded successfully
5. **QC Validation**: ✅ At least 80% of cycles pass QC (≥9 out of 11)
6. **Feature Extraction**: ✅ Features extracted for all QC-passed cycles
7. **No Crashes**: ✅ Orchestrator runs to completion without errors

---

## Test Timeline

```
Start Time: 2026-01-17 11:50:26
End Time (estimated): 2026-01-17 13:01:00 (~70 minutes)

Cycle Schedule:
  Cycle 1:  11:50 - 11:53 (recording) + processing
  Cycle 2:  11:56 - 11:59 (recording) + processing
  Cycle 3:  12:02 - 12:05 (recording) + processing
  Cycle 4:  12:08 - 12:11 (recording) + processing
  Cycle 5:  12:14 - 12:17 (recording) + processing
  Cycle 6:  12:20 - 12:23 (recording) + processing
  Cycle 7:  12:26 - 12:29 (recording) + processing
  Cycle 8:  12:32 - 12:35 (recording) + processing
  Cycle 9:  12:38 - 12:41 (recording) + processing
  Cycle 10: 12:44 - 12:47 (recording) + processing
  Cycle 11: 12:50 - 12:53 (recording) + processing
  Complete: ~13:01
```

---

## Monitoring Commands

```bash
# Monitor log in real-time
tail -f logs/continuous/exp_ee578709_orchestrator.log

# Check experiment status via API
curl -s http://localhost:5001/api/continuous/experiments?limit=1 | jq '.experiments[0]'

# Check downloaded files
ls -lh /Users/bernd/rubix44/recordings/

# Check database status
python -c "
from src.db_connection import DatabaseConnection
db = DatabaseConnection(backend='mariadb')
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute('''
        SELECT cycle_number, status, session_id, qc_passed
        FROM recording_cycles
        WHERE experiment_id = 'exp_ee578709'
        ORDER BY cycle_number
    ''')
    for row in cursor.fetchall():
        print(row)
"
```

---

## Current Status

**Test Started**: 2026-01-17 11:50:26
**Orchestrator PID**: 11921
**Log File**: logs/continuous/exp_ee578709_orchestrator.log

**Cycle 1 Status**:
- ❌ Recording start FAILED (404 error - possibly transient)
- Waiting for Cycle 2 at 11:56:26

**Updates**:
- Fixed playback file to 'exp6 - 3 noise.wav' (valid file on server)
- Monitoring log for next cycle attempt

---

## Results (To Be Updated)

| Cycle | Recording | Metadata | Download | QC | Features | Notes |
|-------|-----------|----------|----------|----|---------  |-------|
| 1     | ❌        | -        | -        | -  | -        | 404 error (transient?) |
| 2     | 🔄        | -        | -        | -  | -        | Waiting... |
| 3     | -         | -        | -        | -  | -        | |
| 4     | -         | -        | -        | -  | -        | |
| 5     | -         | -        | -        | -  | -        | |
| 6     | -         | -        | -        | -  | -        | |
| 7     | -         | -        | -        | -  | -        | |
| 8     | -         | -        | -        | -  | -        | |
| 9     | -         | -        | -        | -  | -        | |
| 10    | -         | -        | -        | -  | -        | |
| 11    | -         | -        | -        | -  | -        | |

---

## Files Modified for This Test

1. `src/continuous/recording_orchestrator.py`
   - Lines 335-386: Added `download_recording()` method
   - Lines 461-478: Integrated download into workflow
   - Lines 191-219: API response parsing (existing fix)
   - Lines 278-327: Enhanced status handling (existing fix)

2. `src/continuous/auto_qc_validator.py`
   - Lines 81-90: StereoChannelProcessor config fix (existing)

3. `web/app.py`
   - Lines 1077-1079: Beaker role defaults (existing fix)

4. Database
   - continuous_experiments table: Updated playback_file for exp_ee578709

---

## Next Steps After Test

Once test completes:

1. **Analyze Results**: Check success rate across all stages
2. **Review QC Metrics**: Examine class separation scores
3. **Validate Downloads**: Confirm all files downloaded correctly
4. **Database Verification**: Ensure all metadata populated
5. **Performance Review**: Check timing and resource usage
6. **Documentation Update**: Record any issues or improvements needed

---

## Comparison to Previous Test

**12-Hour Test (exp_8ca89843)**:
- Duration: 19 hours actual (12 hours planned)
- Total cycles: 119
- Recording success: 72/119 (60%) - due to server downtime
- QC success: 0/72 (0%) - all failed due to missing file access

**This Test (exp_ee578709)**:
- Duration: 70 minutes planned
- Total cycles: 11
- Recording success: TBD
- QC success: TBD (expect >80% with file download fix)

The key difference is the new file download functionality that should enable QC and feature extraction to proceed successfully.
