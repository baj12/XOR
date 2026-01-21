# Substance Vocabulary System - Refactoring Documentation

**Date**: 2026-01-04
**Status**: ✅ Complete

## Overview

Refactored the continuous recording system to use a single-vocabulary approach for substances, eliminating redundancy between descriptive names and class labels.

## Motivation

**Previous System** (Redundant):
```yaml
channel_1_source: "Lavender"       # Descriptive name
channel_1_expected_class: 1        # Numeric class (user must remember mapping)
channel_2_source: "Empty"
channel_2_expected_class: 0
faraday_cage_used: false           # Wrong default
```

**Problems**:
- User had to manually maintain consistency between source name and class
- Easy to make mistakes (e.g., "Lavender" with class 0)
- Not extensible to multi-class scenarios
- Misspellings created inconsistency

**New System** (Single Vocabulary):
```yaml
channel_1_substance: "lavender"    # Auto-maps to class 1
channel_2_substance: "empty"       # Auto-maps to class 0
faraday_cage_used: true            # Correct default
```

**Benefits**:
- Single field per channel
- Automatic class resolution
- Controlled vocabulary with validation
- Handles misspellings gracefully
- Extensible to 8+ classes
- Impossible to have inconsistent source/class pairs

## Changes Made

### 1. Database Schema

**New Table**: `substance_vocabulary`
```sql
CREATE TABLE substance_vocabulary (
    id INT AUTO_INCREMENT PRIMARY KEY,
    substance_name VARCHAR(100) NOT NULL UNIQUE,
    class_label INT NOT NULL,
    is_canonical BOOLEAN DEFAULT TRUE,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Populated with 15 substances**:
- Class 0: empty, control, water, air, blank, nothing
- Class 1: lavender, lavendar (misspelling), lavander (misspelling)
- Class 2-7: peppermint, eucalyptus, rosemary, tea_tree, lemon, orange

**Modified Table**: `continuous_experiments`
- Removed columns: `channel_1_source`, `channel_1_expected_class`, `channel_2_source`, `channel_2_expected_class`
- Added columns: `channel_1_substance`, `channel_2_substance`
- Changed default: `faraday_cage_used` → `TRUE`

**Updated View**: `experiment_summary`
- Now includes `channel_1_substance` and `channel_2_substance`
- Removed references to old columns

**SQL Functions**:
```sql
-- Get class for a substance
SELECT get_substance_class('lavender');  -- Returns 1

-- Validate substance
SELECT is_valid_substance('unknown');    -- Returns 0 (false)
```

**Migration Script**: [scripts/migrate_substance_schema.sql](../scripts/migrate_substance_schema.sql)

### 2. Substance Vocabulary Module

**File**: [src/continuous/substance_vocabulary.py](../src/continuous/substance_vocabulary.py)

**Core Functions**:

```python
from continuous.substance_vocabulary import (
    validate_substance,
    get_class_for_substance,
    get_substance_choices,
    normalize_substance_name
)

# Validate and get class
is_valid, error = validate_substance("Lavender")  # (True, None)
class_label = get_class_for_substance("LAVENDER")  # 1

# UI choices
choices = get_substance_choices()
# Returns: [('air', 0, 'Air (Class 0)'), ('lavender', 1, 'Lavender (Class 1)'), ...]
```

**Features**:
- Case-insensitive matching ("Lavender" == "lavender" == "LAVENDER")
- Space normalization ("tea tree" → "tea_tree")
- Misspelling handling (lavendar/lavander → class 1)
- Helpful error messages with suggestions

### 3. API Endpoints

**Modified**: `POST /api/continuous/experiments`

**Old Request**:
```json
{
  "channel_1_source": "Lavender",
  "channel_1_expected_class": 1,
  "channel_2_source": "Empty",
  "channel_2_expected_class": 0,
  "faraday_cage_used": false
}
```

**New Request**:
```json
{
  "channel_1_substance": "lavender",
  "channel_2_substance": "empty",
  "faraday_cage_used": true
}
```

**New Endpoint**: `GET /api/continuous/substances`

Returns available substances for UI dropdowns:

```json
{
  "success": true,
  "substances": [
    {"name": "empty", "class": 0, "display": "Empty (Class 0)"},
    {"name": "lavender", "class": 1, "display": "Lavender (Class 1)"},
    ...
  ]
}
```

**Validation**: API now validates substance names and returns helpful errors:
```json
{
  "success": false,
  "error": "Unknown substance 'unknown'. Did you mean one of: lavender, lavendar?"
}
```

### 4. Orchestrator Changes

**File**: [src/continuous/recording_orchestrator.py](../src/continuous/recording_orchestrator.py)

**Key Update** (lines 73-86):
```python
def load_experiment_config(self):
    # Load from database
    config = cursor.fetchone()

    # Resolve substance → class
    from continuous.substance_vocabulary import get_class_for_substance

    config['channel_1_expected_class'] = get_class_for_substance(config['channel_1_substance'])
    config['channel_2_expected_class'] = get_class_for_substance(config['channel_2_substance'])

    # Backward compatibility: keep source fields for metadata
    config['channel_1_source'] = config['channel_1_substance']
    config['channel_2_source'] = config['channel_2_substance']

    return config
```

**Result**: Rest of orchestrator code works unchanged - it still uses `channel_X_expected_class` internally.

### 5. Web Interface Changes

**Experiment Creation Form** ([web/templates/continuous_experiment.html](../web/templates/continuous_experiment.html)):

**Old UI**:
```html
<input type="text" id="channel1Source" placeholder="Lavender">
<select id="channel1Class">
  <option value="1">Class 1 (Positive)</option>
  <option value="0">Class 0 (Negative)</option>
</select>
```

**New UI**:
```html
<select id="channel1Substance">
  <option value="">Select substance...</option>
  <!-- Populated dynamically from /api/continuous/substances -->
  <option value="lavender">Lavender (Class 1)</option>
  <option value="empty">Empty (Class 0)</option>
  ...
</select>
<div class="form-text">Class automatically determined</div>
```

**JavaScript Changes**:
- Added `loadSubstances()` function to fetch vocabulary from API
- Removed manual class selection fields
- Updated form submission to send `channel_X_substance` instead of `source`/`class`
- Faraday cage checkbox now checked by default

**Dashboard** ([web/templates/continuous_dashboard.html](../web/templates/continuous_dashboard.html)):
- Displays substance names directly (e.g., "lavender", "empty")
- Removed class label display (redundant with vocabulary system)

## Migration Process

1. **Backup existing data**: Existing experiments automatically migrated
2. **Run migration script**:
   ```bash
   python scripts/run_substance_migration.py
   ```
3. **Verification**:
   ```sql
   SELECT * FROM substance_vocabulary;
   SELECT experiment_id, channel_1_substance, channel_2_substance
   FROM continuous_experiments;
   ```

## Extending the Vocabulary

To add new substances:

**Option 1: Database Insert**
```sql
INSERT INTO substance_vocabulary (substance_name, class_label, is_canonical, description)
VALUES ('rosewood', 8, TRUE, 'Rosewood essential oil');
```

**Option 2: Python Runtime**
```python
from continuous.substance_vocabulary import add_substance

add_substance('rosewood', class_label=8)
```

**Option 3: Update Module** (persistent)

Edit [src/continuous/substance_vocabulary.py](../src/continuous/substance_vocabulary.py):
```python
SUBSTANCE_VOCABULARY = {
    # ... existing entries ...
    'rosewood': 8,
}
```

## Testing

**Test API Endpoint**:
```bash
curl http://localhost:5001/api/continuous/substances
```

**Test Validation**:
```python
from continuous.substance_vocabulary import validate_substance

is_valid, error = validate_substance("lavender")
# (True, None)

is_valid, error = validate_substance("unknown")
# (False, "Unknown substance 'unknown'. Valid options: ...")
```

**Test Experiment Creation**:
```bash
curl -X POST http://localhost:5001/api/continuous/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "Test Vocabulary",
    "target_duration_weeks": 0.01,
    "playback_file": "elvisBlackA.wav",
    "channel_1_substance": "lavender",
    "channel_2_substance": "empty"
  }'
```

## Files Modified

| File | Changes |
|------|---------|
| `scripts/migrate_substance_schema.sql` | NEW - Database migration |
| `scripts/run_substance_migration.py` | NEW - Migration runner |
| `src/continuous/substance_vocabulary.py` | NEW - Vocabulary module |
| `src/continuous/recording_orchestrator.py` | Modified - Auto-resolve classes |
| `web/app.py` | Modified - Updated API endpoints |
| `web/templates/continuous_experiment.html` | Modified - New substance UI |
| `web/templates/continuous_dashboard.html` | Modified - Display substances |
| `CONTINUOUS_IMPLEMENTATION_STATUS.md` | Updated - Document refactoring |

## Breaking Changes

⚠️ **No Backward Compatibility**

Old API requests with `channel_X_source` and `channel_X_expected_class` will be rejected.

**Migration for API Clients**:
```python
# OLD
payload = {
    "channel_1_source": "Lavender",
    "channel_1_expected_class": 1
}

# NEW
payload = {
    "channel_1_substance": "lavender"
}
```

## Benefits Realized

✅ **Simplified Configuration**: Users only need to know substance names, not class mappings
✅ **Reduced Errors**: Impossible to have mismatched source/class pairs
✅ **Better UX**: Dropdown selection instead of free-text + manual class
✅ **Extensible**: Easy to add new substances and multi-class scenarios
✅ **Misspelling Tolerant**: Common mistakes handled gracefully
✅ **Validated**: API enforces vocabulary, preventing invalid data
✅ **Faraday Default Fixed**: Now correctly defaults to TRUE

## Future Enhancements

- [ ] Add substance aliases via UI (not just code)
- [ ] Substance usage statistics (most common substances)
- [ ] Multi-language support (e.g., "lavanda" for Spanish)
- [ ] Substance properties (e.g., concentration, batch number)
- [ ] Auto-suggest based on previous experiments
