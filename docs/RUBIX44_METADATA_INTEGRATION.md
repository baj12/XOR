# Rubix44 Metadata Integration - Implementation Summary

**Date:** 2026-01-04
**Status:** ✅ Complete (Priority 1 of Phase 2)

## Overview

The Rubix44 data provider has been updated to integrate with the MariaDB metadata system. Recordings are now only processed if they have complete metadata and quality approval, and class labels are taken directly from metadata instead of being hardcoded.

## Changes Made

### 1. Updated `src/continuous/rubix44_data_provider.py`

#### Added Dependencies
```python
from db_connection import DatabaseConnection
```

#### Constructor Updated
Added `metadata_backend` parameter (default: 'mariadb'):
```python
def __init__(self, ..., metadata_backend: str = 'mariadb'):
    ...
    self.metadata_db = DatabaseConnection(backend=metadata_backend)
```

#### New Method: `_get_metadata()`
Queries MariaDB for recording metadata and validates completeness/approval:

```python
def _get_metadata(self, session_id: str) -> Optional[Dict]:
    """
    Retrieve metadata for a recording session from MariaDB.

    Returns:
        Dictionary with metadata if found and approved, None otherwise
    """
    with self.metadata_db.get_connection() as conn:
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT metadata_complete, quality_approved,
                   channel_1_expected_class, channel_2_expected_class,
                   channel_1_source, channel_2_source
            FROM recording_sessions
            WHERE session_id = %s
        """
        cursor.execute(query, (session_id,))
        metadata = cursor.fetchone()

        # Validation checks
        if not metadata:
            return None
        if not metadata.get('metadata_complete'):
            return None
        if not metadata.get('quality_approved'):
            return None

        return metadata
```

**Key Features:**
- Returns `None` if metadata not found
- Returns `None` if `metadata_complete != TRUE`
- Returns `None` if `quality_approved != TRUE`
- Logs channel configuration for verification

#### New Method: `_update_processing_flags()`
Updates processing status in MariaDB after successful feature extraction:

```python
def _update_processing_flags(self, session_id: str,
                             imported: bool = True,
                             processed: bool = True):
    """
    Update processing status flags in MariaDB.
    """
    with self.metadata_db.get_connection() as conn:
        cursor = conn.cursor()

        query = """
            UPDATE recording_sessions
            SET imported_to_features_db = %s,
                processed_for_training = %s
            WHERE session_id = %s
        """
        cursor.execute(query, (imported, processed, session_id))
        conn.commit()
```

**Purpose:**
- Marks recordings as processed to avoid duplicate work
- Tracks import status for monitoring
- Enables downstream pipeline to identify ready recordings

#### Updated Method: `_process_session()`
Major changes to integrate metadata validation and use metadata-specified labels:

**Before:**
```python
def _process_session(self, session: Dict) -> bool:
    # ... download file ...

    # HARDCODED LABELS
    X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
        self.processor.process_stereo_file(local_path)

    # Labels were always: left=1 (positive), right=0 (negative)
```

**After:**
```python
def _process_session(self, session: Dict) -> bool:
    session_id = session['id']

    # CHECK METADATA FIRST
    metadata = self._get_metadata(session_id)
    if not metadata:
        logger.info(f"Skipping {session_id}: no valid metadata")
        return False

    # ... download file ...

    # GET LABELS FROM METADATA
    ch1_class = int(metadata['channel_1_expected_class'])
    ch2_class = int(metadata['channel_2_expected_class'])

    logger.info(f"Using labels from metadata: Ch1={ch1_class}, Ch2={ch2_class}")

    # PROCESS WITH METADATA LABELS
    X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
        self.processor.process_stereo_file(
            local_path,
            positive_label=ch1_class,
            negative_label=ch2_class
        )

    # ... store features ...

    # UPDATE PROCESSING FLAGS
    self._update_processing_flags(session_id, imported=True, processed=True)

    return True
```

**Key Changes:**
1. **Early metadata check** - Skips processing if metadata not ready
2. **Dynamic labels** - Uses `channel_1_expected_class` and `channel_2_expected_class` from metadata
3. **Processing tracking** - Updates flags in MariaDB after success
4. **Better logging** - Shows which labels are being used

### 2. Updated `src/continuous/stereo_channel_processor.py`

#### Updated Method: `process_stereo_file()`
Added support for custom class labels:

**Before:**
```python
def process_stereo_file(self, wav_path: Path,
                       max_samples_per_channel: Optional[int] = None):
    # ... extract features ...

    # HARDCODED LABELS
    y_left = np.ones(len(X_left), dtype=int)    # Always 1
    y_right = np.zeros(len(X_right), dtype=int) # Always 0

    return X_left, y_left, X_right, y_right, offsets_left, offsets_right
```

**After:**
```python
def process_stereo_file(self, wav_path: Path,
                       max_samples_per_channel: Optional[int] = None,
                       positive_label: int = 1,
                       negative_label: int = 0):
    """
    Process stereo WAV file with configurable class labels.

    Args:
        positive_label: Label for left channel (default: 1)
        negative_label: Label for right channel (default: 0)
    """
    logger.info(f"Using labels: left={positive_label}, right={negative_label}")

    # ... extract features ...

    # USE PROVIDED LABELS
    y_left = np.full(len(X_left), positive_label, dtype=int)
    y_right = np.full(len(X_right), negative_label, dtype=int)

    return X_left, y_left, X_right, y_right, offsets_left, offsets_right
```

**Improvements:**
- **Backward compatible** - Defaults to 1/0 if not specified
- **Flexible labeling** - Supports any integer labels from metadata
- **Better logging** - Shows which labels are being applied

## Workflow Integration

### Complete Processing Flow

```
1. User records audio via web interface
   ↓
2. User annotates with metadata (channel sources, expected classes, beakers, etc.)
   ↓
3. User marks metadata as complete
   ↓
4. User performs QC and approves quality
   ↓
5. Rubix44DataProvider polls for new recordings (every 5 min)
   ↓
6. For each recording:
   a. Query MariaDB for metadata
   b. Skip if metadata_complete != TRUE or quality_approved != TRUE
   c. Download stereo WAV file
   d. Extract features with metadata-specified labels
   e. Store features in database
   f. Update processing flags in MariaDB
   ↓
7. Incremental trainer uses new features automatically
```

### Example Scenario

**Recording:** `lavender_test_2026-01-04_16-30-00`

**Metadata in MariaDB:**
```sql
session_id: lavender_test_2026-01-04_16-30-00
channel_1_source: Beaker_A
channel_1_expected_class: 1
channel_2_source: Beaker_B (Empty)
channel_2_expected_class: 0
metadata_complete: TRUE
quality_approved: TRUE
```

**Processing:**
```
1. Rubix44DataProvider finds new recording
2. Queries MariaDB → finds metadata
3. Validates: metadata_complete=TRUE, quality_approved=TRUE ✅
4. Downloads: lavender_test_2026-01-04_16-30-00_stereo.wav
5. Processes with labels: left=1, right=0
6. Stores ~7,200 features (120s × 60 samples/sec)
7. Updates flags: imported_to_features_db=TRUE, processed_for_training=TRUE
```

**If metadata was incomplete:**
```
1. Rubix44DataProvider finds recording
2. Queries MariaDB → metadata_complete=FALSE
3. Skips processing ⏭️
4. Recording stays in "Pending Metadata" state
```

## Benefits

### 1. **Prevents Invalid Processing**
- No recordings processed without complete metadata
- No recordings processed without quality approval
- Eliminates risk of incorrect labels

### 2. **Flexible Channel Assignment**
- Users can assign any class to any channel
- Example: Both channels could be positive (class=1) for dual-substance tests
- Example: Both channels could be negative (class=0) for baseline recordings

### 3. **Audit Trail**
- Processing flags track which recordings have been imported
- Prevents duplicate processing
- Enables monitoring and reporting

### 4. **Clean Separation of Concerns**
- Metadata management via web interface
- Automated processing via provider
- Clear workflow stages

## Testing

### Manual Testing Checklist

- [ ] Recording without metadata is skipped
- [ ] Recording with incomplete metadata is skipped
- [ ] Recording with complete metadata but not approved is skipped
- [ ] Recording with complete + approved metadata is processed
- [ ] Correct labels are used from metadata
- [ ] Processing flags are updated after success
- [ ] Non-standard labels work (e.g., both channels class=1)

### Test Case Example

```python
# Create test recording in MariaDB
INSERT INTO recording_sessions (
    session_id,
    channel_1_expected_class,
    channel_2_expected_class,
    metadata_complete,
    quality_approved
) VALUES (
    'test_2026-01-04_10-00-00',
    1,  # Left = positive
    0,  # Right = negative
    TRUE,
    TRUE
);

# Run provider
provider.poll_for_new_recordings()

# Verify in features database
SELECT COUNT(*), label
FROM features
WHERE source_file LIKE '%test_2026-01-04_10-00-00%'
GROUP BY label;
```

Expected result:
```
label | count
------|------
1     | ~3600  (left channel)
0     | ~3600  (right channel)
```

## Configuration

### Orchestrator Config Update

When using the orchestrator, ensure rubix44 provider initialization includes metadata backend:

```yaml
# config/continuous_learning_config.yaml
rubix44:
  api_url: "http://10.0.0.58:5000"
  download_dir: "data/continuous/recordings"
  output_prefix_filter: null  # Process all recordings
  cleanup_after_processing: false
  metadata_backend: "mariadb"  # NEW: Use MariaDB for metadata
```

### Direct Usage

```python
from continuous.rubix44_data_provider import Rubix44DataProvider
from continuous.stereo_channel_processor import StereoChannelProcessor
from continuous.feature_database import FeatureDatabase

# Initialize components
processor = StereoChannelProcessor(config)
database = FeatureDatabase(backend='mariadb')

# Create provider with metadata integration
provider = Rubix44DataProvider(
    api_url="http://10.0.0.58:5000",
    download_dir=Path("data/continuous/recordings"),
    processor=processor,
    database=database,
    metadata_backend='mariadb'  # Enable metadata checking
)

# Poll for recordings (only processes approved ones)
processed_count = provider.poll_for_new_recordings()
print(f"Processed {processed_count} recordings")
```

## Migration Notes

### Existing Deployments

If you have an existing continuous learning system running:

1. **Stop the orchestrator/provider**
   ```bash
   # Find and stop running process
   ps aux | grep orchestrator
   kill <pid>
   ```

2. **Update the code**
   ```bash
   git pull
   # Files updated:
   # - src/continuous/rubix44_data_provider.py
   # - src/continuous/stereo_channel_processor.py
   ```

3. **No database migration needed**
   - Schema already includes required fields
   - Existing recordings remain unchanged

4. **Restart with new behavior**
   ```bash
   python -m src.continuous.orchestrator \
       --config config/continuous_learning_config.yaml \
       --db data/continuous/features.db \
       --model-dir models/continuous \
       --log INFO
   ```

### Backward Compatibility

The changes are **fully backward compatible**:

- `positive_label` and `negative_label` default to 1 and 0
- Code that doesn't specify labels continues to work
- Existing tests pass without modification
- Old recordings can still be reprocessed

## Monitoring

### Check Processing Status

```sql
-- See which recordings have been processed
SELECT session_id, metadata_complete, quality_approved,
       imported_to_features_db, processed_for_training
FROM recording_sessions
ORDER BY recording_date DESC
LIMIT 10;
```

### Check Pending Recordings

```sql
-- Recordings waiting for metadata
SELECT COUNT(*) as pending_metadata
FROM recording_sessions
WHERE metadata_complete = FALSE;

-- Recordings waiting for QC
SELECT COUNT(*) as pending_qc
FROM recording_sessions
WHERE metadata_complete = TRUE
  AND quality_approved IS NULL;

-- Recordings ready but not yet processed
SELECT COUNT(*) as ready_to_process
FROM recording_sessions
WHERE metadata_complete = TRUE
  AND quality_approved = TRUE
  AND imported_to_features_db = FALSE;
```

### Logs

Look for these log messages:

**Successful processing:**
```
INFO - Session lavender_test_2026-01-04_16-30-00: metadata validated - Ch1=Beaker_A(class=1), Ch2=Beaker_B(class=0)
INFO - Using labels from metadata: Ch1=1, Ch2=0
INFO - Storing 3600 channel 1 (class=1) samples...
INFO - Storing 3600 channel 2 (class=0) samples...
INFO - Updated processing flags for lavender_test_2026-01-04_16-30-00: imported=True, processed=True
```

**Skipped (no metadata):**
```
WARNING - No metadata found for session test_2026-01-04_10-00-00
INFO - Skipping session test_2026-01-04_10-00-00: no valid metadata
```

**Skipped (metadata incomplete):**
```
INFO - Session test_2026-01-04_10-00-00: metadata not complete, skipping
```

**Skipped (not approved):**
```
INFO - Session test_2026-01-04_10-00-00: quality not approved, skipping
```

## Next Steps

This completes **Priority 1** of Phase 2. The next priority is:

### Priority 2: QC Visualization

Create quality control interface with:
- UMAP/t-SNE/PCA visualizations
- Class separation metrics
- Approve/Reject workflow
- Integration with quality_approved flag

See [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) for details.

## Related Documentation

- [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) - Overall project status
- [QUICKSTART_RECORDING.md](../QUICKSTART_RECORDING.md) - User guide for recording system
- [MARIADB_SCHEMA.md](MARIADB_SCHEMA.md) - Database schema reference
- [WEB_INTERFACE_GUIDE.md](WEB_INTERFACE_GUIDE.md) - Web interface documentation

---

**Implementation Complete:** ✅ 2026-01-04
**Tested:** Manual testing required
**Production Ready:** Yes (pending testing)
