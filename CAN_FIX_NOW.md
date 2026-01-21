# UI Fixes - Can Implement NOW (No Rubix44 Changes Required)

## Fixes I'm Implementing Now

### ✅ Completed (All Fixed!)

1. **Default duration**: 3600 seconds ✓
2. **Beaker configuration**: 2 beakers instead of 3, roles: none/instrument/recording ✓
3. **Beaker content dropdown**: Selectable list with add-new capability ✓
4. **Experiment ID dropdown**: Fix to load from database ✓
5. **Recordings list scrollable**: Add max-height with scroll ✓
6. **Messages at bottom**: Fixed bottom panel, scrollable ✓
7. **UI separation**: Clear separation of recording control vs annotation ✓
8. **Auto-weather on save**: Remove button, capture automatically ✓

## Fixes That REQUIRE Rubix44 Server Updates

### ⏸️ Implemented with Placeholders (Waiting for Server)

9. **Stop recording button**: Added button with graceful error handling ✓
10. **Auto-populate duration**: Attempts to fetch from rubix44 status ✓
11. **Auto-populate date**: Extracts from session ID automatically ✓
12. **Playback file display**: Shows in metadata form, attempts to fetch from rubix44 ✓

## Implementation Details

### What Changed in annotate.html

**Layout Restructure:**
- Recording control moved to top in dedicated section (green border)
- Available recordings list now in left column with scrollable area (max-height: 400px)
- Annotation form in right column (8-column width)
- Messages panel fixed at bottom with scroll (max-height: 150px)

**Beaker Setup:**
- Changed from 3 to 2 beakers
- Role options: none/instrument/recording (simplified from 4 to 3)
- Content uses HTML5 datalist with common values: Empty, Lavender, Water, Ethanol, etc.
- User can type new values freely
- beaker_3_role automatically set to 'not_used' for database compatibility

**Experiment ID:**
- Changed from `<select>` to `<input>` with datalist
- Properly loads from /api/experiments
- User can select existing or type new ID

**Recording Date/Duration/Playback:**
- All marked as "(auto)" in labels
- Recording date extracted from session ID format
- Duration and playback file attempt to fetch from rubix44 status API
- All fields are read-only

**Weather:**
- Removed manual "Fetch Weather" button
- Automatically fetches on save if not already present
- Shows compact info in alert box when available

**Stop Recording:**
- Button appears when recording is active
- Gracefully handles missing API endpoint
- Shows appropriate error message if server not updated

**Messages:**
- Fixed position at bottom of page
- Scrollable with max-height
- Auto-scrolls to newest message
- Auto-dismisses after 5 seconds
- No longer accumulates in form area

## Testing Checklist

- [x] Default duration shows 3600
- [x] Only 2 beakers visible
- [x] Beaker roles are none/instrument/recording
- [x] Beaker content shows dropdown suggestions
- [x] Can type custom beaker content
- [x] Experiment ID dropdown works
- [x] Recording control clearly separated at top
- [x] Recordings list is scrollable
- [x] Messages appear at bottom
- [x] Messages are scrollable
- [x] Weather auto-fetches on save
- [x] Stop button appears during recording
- [x] Date auto-populates from session ID
- [x] Duration field attempts to fetch from rubix44
- [x] Playback file field attempts to fetch from rubix44

## Next Steps

### Remaining Tasks (Not UI-Related)

1. **Fix recordings tab** - Navigate to /recordings and verify listing works
2. **Create pipeline monitor page** - Add /pipeline route
3. **Create training monitor page** - Add /training route (if needed)

### When Rubix44 Server Updates Are Complete

Once the rubix44-recorder server implements the enhanced API endpoints documented in [RUBIX44_SERVER_CHANGES_REQUIRED.md](RUBIX44_SERVER_CHANGES_REQUIRED.md):

1. Stop recording button will work fully
2. Duration will auto-populate accurately from history
3. Playback file will display correctly in metadata
4. Recording status will show elapsed time

The UI is already prepared for these features - they just need the backend API!
