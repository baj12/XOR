# Rubix44 Recording Management Guide

**Date**: 2026-01-11
**Status**: ✅ Fully Operational

## Overview

This guide documents the complete recording management workflow for rubix44-recorder, including:
1. Testing API endpoints
2. Transferring recordings to CIH storage
3. Storing metadata in MariaDB
4. Deleting recordings from server

## System Components

### 1. Rubix44-Recorder API (10.0.0.58:5000)
- **Location**: Windows server at 10.0.0.58
- **Purpose**: Records stereo audio and provides REST API
- **Storage**: Local `recordings/` directory on Windows server

### 2. CIH Storage (/Volumes/CIH/mora/moraWav/rubix/)
- **Location**: Network share mounted on Mac
- **Purpose**: Long-term storage for recordings
- **Format**: WAV files (stereo, ch1, ch2)

### 3. MariaDB Database (10.0.0.103)
- **Database**: `xor_project`
- **Table**: `recording_sessions`
- **Purpose**: Metadata tracking and experiment annotations

## API Endpoints Tested

All v1.1.0 API endpoints are now operational and tested:

### ✅ Health & Configuration
- `GET /api/v1/health` - Health check
- `GET /api/v1/config` - Server configuration
- `GET /api/v1/storage/config` - Storage server settings
- `PUT /api/v1/storage/config` - Update storage settings

### ✅ Devices
- `GET /api/v1/devices` - List audio devices
- `GET /api/v1/devices/rubix` - Verify Rubix44 connected

### ✅ Recordings
- `GET /api/v1/recordings/history` - List all recordings (with v1.1.0 enhanced metadata)
- `GET /api/v1/recordings/status` - Current recording status
- `GET /api/v1/recordings/{filename}` - Download recording file
- `POST /api/v1/recordings/delete` - Delete recording session

### ✅ Playback
- `GET /api/v1/playback-files` - List available stimulus files

## Transfer Workflow

### Script: `transfer_rubix_recordings.py`

**Location**: `/Users/bernd/python/XOR/scripts/transfer_rubix_recordings.py`

**Features**:
- Downloads recordings from API to CIH storage
- Stores metadata in MariaDB `recording_sessions` table
- Optionally deletes from server after transfer
- Supports single session or batch transfer
- Dry-run mode for testing

### Usage Examples

#### 1. Dry Run (Preview What Would Transfer)
```bash
python scripts/transfer_rubix_recordings.py --dry-run
```

Output shows what would be transferred without actually doing it.

#### 2. Transfer All Recordings
```bash
python scripts/transfer_rubix_recordings.py
```

This will:
- Fetch all recordings from API
- Download to `/Volumes/CIH/mora/moraWav/rubix/`
- Store metadata in MariaDB
- Skip files that already exist locally

#### 3. Transfer Specific Recording
```bash
python scripts/transfer_rubix_recordings.py \
    --session-id recording_2026-01-04_18-11-06
```

#### 4. Transfer and Delete from Server
```bash
python scripts/transfer_rubix_recordings.py \
    --delete-after-transfer
```

**⚠️ Warning**: Only use `--delete-after-transfer` after verifying files transferred successfully!

#### 5. Custom Destination
```bash
python scripts/transfer_rubix_recordings.py \
    --destination /path/to/custom/location
```

## Database Schema

Metadata is stored in the `recording_sessions` table with these key fields:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | VARCHAR(255) | Unique session ID from rubix44 |
| `recording_date` | DATETIME | Recording start time |
| `duration_seconds` | FLOAT | Recording duration |
| `sample_rate` | INT | Audio sample rate (44100) |
| `stereo_filename` | VARCHAR(500) | Stereo WAV filename |
| `ch1_filename` | VARCHAR(500) | Channel 1 filename |
| `ch2_filename` | VARCHAR(500) | Channel 2 filename |
| `file_size_bytes` | BIGINT | Total file size |
| `created_at` | TIMESTAMP | When record was created |

**Additional fields available for annotations**:
- Channel configuration (channel_1_source, channel_2_source, expected class labels)
- Beaker setup (3 beakers with roles and contents)
- Experimental conditions (Faraday cage, descriptions, researcher name)
- Weather data (automatically captured for Viroflay, France)
- Processing status flags (metadata_complete, quality_approved, processed_for_training)
- Comments and tags

## Testing Results

### API Endpoint Tests (2026-01-11)

#### Health Check ✅
```bash
curl http://10.0.0.58:5000/api/v1/health
```
Response: `{"status": "healthy", "service": "Rubix Recorder API"}`

#### Recording History ✅
```bash
curl http://10.0.0.58:5000/api/v1/recordings/history
```
Returns 15 recordings with full v1.1.0 metadata including:
- duration_seconds
- playback_file
- sample_rate
- start_time / end_time
- file paths and sizes

#### Delete Recording ✅
```bash
curl -X POST http://10.0.0.58:5000/api/v1/recordings/delete \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "test_recording_2026-01-04_11-50-13"}'
```
Response:
```json
{
  "success": true,
  "session_id": "test_recording_2026-01-04_11-50-13",
  "deleted_count": 3,
  "deleted_files": [
    "test_recording_2026-01-04_11-50-13_stereo.wav",
    "test_recording_2026-01-04_11-50-13_ch1.wav",
    "test_recording_2026-01-04_11-50-13_ch2.wav"
  ],
  "failed_files": []
}
```

### Transfer Tests (2026-01-11)

#### Test Recording Transfer ✅
```bash
python scripts/transfer_rubix_recordings.py \
    --session-id test_recording_2026-01-04_18-11-06
```

Results:
- ✅ 3 files downloaded (5.0 MB stereo + 2.5 MB × 2 channels = 10 MB total)
- ✅ Files stored in `/Volumes/CIH/mora/moraWav/rubix/`
- ✅ Metadata inserted into MariaDB
- ✅ All fields populated correctly

#### Batch Transfer ✅
```bash
python scripts/transfer_rubix_recordings.py
```

Results (as of 2026-01-11 15:30):
- ✅ 9 recordings transferred successfully
- ✅ Total data: ~4.3 GB (including large 1-hour continuous recordings)
- ✅ All metadata stored in database
- ✅ No errors

## Current Recordings in Storage

**Location**: `/Volumes/CIH/mora/moraWav/rubix/`

| Recording Type | Count | Total Size | Notes |
|---------------|-------|------------|-------|
| Test recordings | 3 | ~30 MB | Short test files (10-60 seconds) |
| Continuous learning | 4 | ~4.2 GB | 1-hour sessions from continuous experiment |
| Regular recordings | 2 | ~37 MB | Various durations (10-100 seconds) |

**Total**: 9 recording sessions, ~4.3 GB

## Maintenance Workflows

### Daily: Transfer New Recordings

Run this daily (or set up as cron job):

```bash
# Transfer new recordings
python scripts/transfer_rubix_recordings.py

# Optionally delete from server after verifying transfer
# python scripts/transfer_rubix_recordings.py --delete-after-transfer
```

### Weekly: Verify Database Consistency

```python
import sys
sys.path.insert(0, 'src')
from db_connection import DatabaseConnection

db = DatabaseConnection(backend='mariadb')
with db.get_connection() as conn:
    cursor = conn.cursor(dictionary=True)

    # Check for recordings
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(duration_seconds) as total_duration,
               SUM(file_size_bytes) / 1024 / 1024 / 1024 as total_gb
        FROM recording_sessions
    """)
    stats = cursor.fetchone()
    print(f"Total recordings: {stats['total']}")
    print(f"Total duration: {stats['total_duration']/3600:.1f} hours")
    print(f"Total size: {stats['total_gb']:.2f} GB")
```

### Monthly: Archive Old Recordings

For recordings older than 3 months, consider:
1. Verifying database entries are complete
2. Compressing WAV files (gzip)
3. Moving to cold storage
4. Updating database with new file paths

## Integration with Continuous Learning

The transfer script is designed to work seamlessly with the continuous learning pipeline:

### 1. Recording Sessions Table
- Stores all recording metadata
- Tracks processing status flags
- Links to experiment descriptions

### 2. Features Database
- Features extracted from WAV files
- References recording_sessions via session_id
- Used for training models

### 3. Workflow
```
Rubix44 Records → Transfer Script → CIH Storage + MariaDB
                                              ↓
                                    Continuous Learning Pipeline
                                              ↓
                                        Feature Extraction
                                              ↓
                                    Model Training & Inference
```

## Troubleshooting

### Issue: CIH Volume Not Mounted
**Error**: `Destination parent directory does not exist`

**Solution**:
```bash
# Check if mounted
ls /Volumes/CIH/mora/moraWav/rubix/

# If not mounted, mount the network share
# (Instructions depend on your network setup)
```

### Issue: Database Connection Failed
**Error**: `Unable to connect to database`

**Solution**:
```bash
# Check MariaDB is running
ping 10.0.0.103

# Verify credentials in /Users/bernd/python/.env
cat /Users/bernd/python/.env | grep MARIADB
```

### Issue: API Not Responding
**Error**: `Connection refused` or `Timeout`

**Solution**:
```bash
# Check rubix44-recorder is running
curl http://10.0.0.58:5000/api/v1/health

# If not responding, check Windows server at 10.0.0.58
```

### Issue: Files Already Exist
**Behavior**: Script skips existing files

**Solution**: This is expected behavior. The script will:
- Skip downloading if file exists
- Update database metadata
- Log "File already exists" message

To force re-download, delete the local files first.

## Security Considerations

### API Access
- API currently has no authentication
- Only accessible on local network (10.0.0.x)
- For production, implement token-based auth

### Database Credentials
- Stored in `/Users/bernd/python/.env`
- Not committed to git
- Uses read/write permissions for xor_project database

### File Storage
- CIH volume permissions: rwxrwxrwx (full access)
- Consider restricting permissions for production use

## Future Enhancements

### Priority 1: Automation
- [ ] Set up cron job for daily transfers
- [ ] Email notifications on transfer completion/errors
- [ ] Automatic cleanup of server after successful transfer

### Priority 2: Validation
- [ ] Verify WAV file integrity after transfer (checksum)
- [ ] Compare file sizes before/after
- [ ] Automated QC on transferred files

### Priority 3: Compression
- [ ] Compress old recordings (gzip)
- [ ] Update database with compressed file paths
- [ ] Transparent decompression for analysis

### Priority 4: Remote Storage
- [ ] Implement rubix44-recorder's storage_server transfer
- [ ] Use SCP/rsync for direct server-to-storage transfer
- [ ] Eliminate intermediate Mac transfer step

## Related Documentation

- **API Documentation**: [RUBIX44_SERVER_UPDATE.md](RUBIX44_SERVER_UPDATE.md) - Complete API specification
- **API Integration**: [RUBIX44_API_INTEGRATION_REVIEW.md](RUBIX44_API_INTEGRATION_REVIEW.md) - Gap analysis
- **v1.1.0 Upgrade**: [RUBIX44_API_V1.1.0_UPGRADE.md](RUBIX44_API_V1.1.0_UPGRADE.md) - Implementation summary
- **Metadata Integration**: [RUBIX44_METADATA_INTEGRATION.md](RUBIX44_METADATA_INTEGRATION.md) - Database schema
- **Database Schema**: [MARIADB_SCHEMA.md](MARIADB_SCHEMA.md) - Complete schema documentation

## Summary

✅ **All systems operational**:
- Rubix44-recorder API running at 10.0.0.58:5000
- CIH storage mounted and accessible
- MariaDB database configured and tested
- Transfer script working correctly
- 9 recordings transferred and cataloged

✅ **API endpoints tested and working**:
- Health checks
- Recording history with v1.1.0 metadata
- File downloads
- Recording deletion

✅ **Database integration complete**:
- Metadata stored in recording_sessions table
- All fields populated correctly
- Ready for continuous learning pipeline

**Next steps**: Set up automated daily transfers and integrate with continuous learning feature extraction.
