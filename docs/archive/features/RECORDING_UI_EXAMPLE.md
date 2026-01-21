# Recording Time Estimation - UI Examples

## Before (Old UI)
```
┌─────────────────────────────────────────┐
│ Recording Status                        │
├─────────────────────────────────────────┤
│ ⚠️ Recording in progress...             │
│ Session: rec_20260112_143022            │
│ Elapsed: 1800s                          │
│ ⏳ Loading...                           │
└─────────────────────────────────────────┘
```

**Issues:**
- Raw seconds are hard to interpret (1800s = ?)
- No indication of how much time is remaining
- No visual progress indicator
- Hard to estimate when recording will complete

## After (New UI)

### Example 1: Recording Starting (5% complete)
```
┌─────────────────────────────────────────┐
│ Recording Status                        │
├─────────────────────────────────────────┤
│ 🔴 Recording in progress...             │
│ Session: rec_20260112_143022            │
│                                         │
│ 🕐 Elapsed: 3m 0s    ⏳ Remaining: 57m 0s│
│ Total duration: 1h 0m                   │
│                                         │
│ ██░░░░░░░░░░░░░░░░░░░░░░ 5.0%         │
│                                         │
│ ⌛ Loading...                           │
└─────────────────────────────────────────┘
```

### Example 2: Recording Mid-way (50% complete)
```
┌─────────────────────────────────────────┐
│ Recording Status                        │
├─────────────────────────────────────────┤
│ 🔴 Recording in progress...             │
│ Session: rec_20260112_143022            │
│                                         │
│ 🕐 Elapsed: 30m 0s   ⏳ Remaining: 30m 0s│
│ Total duration: 1h 0m                   │
│                                         │
│ ████████████░░░░░░░░░░░░ 50.0%         │
│                                         │
│ ⌛ Loading...                           │
└─────────────────────────────────────────┘
```

### Example 3: Recording Almost Done (95% complete)
```
┌─────────────────────────────────────────┐
│ Recording Status                        │
├─────────────────────────────────────────┤
│ 🔴 Recording in progress...             │
│ Session: rec_20260112_143022            │
│                                         │
│ 🕐 Elapsed: 57m 0s   ⏳ Remaining: 3m 0s │
│ Total duration: 1h 0m                   │
│                                         │
│ ███████████████████████░ 95.0%         │
│                                         │
│ ⌛ Loading...                           │
└─────────────────────────────────────────┘
```

### Example 4: Long Recording (Multi-hour)
```
┌─────────────────────────────────────────┐
│ Recording Status                        │
├─────────────────────────────────────────┤
│ 🔴 Recording in progress...             │
│ Session: exp_continuous_001             │
│                                         │
│ 🕐 Elapsed: 2h 15m   ⏳ Remaining: 1h 45m│
│ Total duration: 4h 0m                   │
│                                         │
│ ██████████████░░░░░░░░░░ 56.3%         │
│                                         │
│ ⌛ Loading...                           │
└─────────────────────────────────────────┘
```

## Benefits

✅ **Easy to understand** - Human-readable time format (1h 30m instead of 5400s)
✅ **Predictable** - Shows exactly how much time is remaining
✅ **Visual feedback** - Animated progress bar provides at-a-glance status
✅ **Professional** - Matches industry-standard UI patterns
✅ **No extra clicks** - Information is immediately visible

## Technical Implementation

### Time Format Rules
- `< 60 seconds` → "45s"
- `< 1 hour` → "15m 30s"
- `< 24 hours` → "2h 15m"
- `≥ 24 hours` → "2d 12h" or "1d"

### Progress Bar
- **Color:** Warning (yellow/orange) during recording
- **Animation:** Striped and animated
- **Height:** 25px (comfortable to read percentage)
- **Update rate:** Every 5 seconds (automatic polling)

### Icons Used (Bootstrap Icons)
- 🔴 `bi-record-circle-fill` - Recording indicator (red)
- 🕐 `bi-clock-history` - Elapsed time icon
- ⏳ `bi-hourglass-split` - Remaining time icon
- ⌛ `spinner-border` - Loading animation

## User Experience Flow

1. **User clicks "Start Recording"**
   - Form is disabled
   - Status panel becomes visible
   - Initial display: "0s elapsed, full duration remaining"

2. **Recording progresses** (auto-updates every 5s)
   - Elapsed time increases: "5s → 10s → 15s..."
   - Remaining time decreases: "1h 0m → 59m 55s → 59m 50s..."
   - Progress bar fills: "0% → 1% → 2%..."

3. **Recording completes**
   - Status changes from "recording" to "idle"
   - Success message: "Recording completed! Refreshing list..."
   - Status panel hides
   - Recording list refreshes automatically

4. **User can stop early** (optional)
   - "Stop Recording Now" button remains visible
   - Stops recording before completion time

## Accessibility

- **Screen readers:** Progress bar includes ARIA attributes
  - `aria-valuenow` - Current percentage
  - `aria-valuemin="0"` - Minimum value
  - `aria-valuemax="100"` - Maximum value
  - `role="progressbar"` - Semantic meaning

- **Keyboard navigation:** All controls remain keyboard accessible

- **Color contrast:** Progress bar text is readable on yellow/orange background

## Browser Compatibility

✅ All modern browsers (Chrome, Firefox, Safari, Edge)
✅ Bootstrap 5 progress bars (already in use)
✅ No special CSS or JavaScript features required
✅ Works on mobile devices (responsive design)
