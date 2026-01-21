# Recording Management System - Quick Start Guide

Complete workflow for recording, annotating, and processing audio data for continuous learning.

---

## 🚀 Setup (One-Time)

### 1. Initialize Database

```bash
# Activate conda environment
source /Users/bernd/miniconda3/bin/activate xorProject

# Initialize MariaDB schema
python scripts/init_recording_schema.py
```

**Expected output:**
```
✓ All recording management tables created successfully
✓ Schema verification completed
```

### 2. Set Environment Variables (Optional)

```bash
# Weather API key (optional - will use free Open-Meteo as fallback)
export OPENWEATHER_API_KEY="your-api-key-here"

# Rubix44 API URL (default is already configured)
export RUBIX44_URL="http://10.0.0.58:5000"
```

### 3. Start Web Interface

```bash
# Start on default port 5001
python web/app.py

# Or custom port
python web/app.py --port 8080
```

**Access:** http://localhost:5001

---

## 📝 Workflow: Record → Annotate → Process

### Method 1: Start Recording from Web Interface (Recommended)

1. **Open web interface:** http://localhost:5001/annotate

2. **Start New Recording:**
   - Select playback file from dropdown
   - Set duration (seconds)
   - Set output prefix (e.g., "experiment_001")
   - Click **"Start Recording"**

3. **Wait for completion** (status updates automatically)

4. **Recording appears** in left panel when done

5. **Click on recording** to select it for annotation

6. **Fill metadata form:**

   **Channel Configuration:**
   - Channel 1 (Left) Source: e.g., "Beaker_A"
   - Channel 1 Expected Class: 1 (Positive) or 0 (Negative)
   - Channel 2 (Right) Source: e.g., "Beaker_B"
   - Channel 2 Expected Class: 0 or 1

   **Beaker Setup (3 beakers):**
   - Beaker 1 Role: Recording/Instrument/Empty/Not Used
   - Beaker 1 Content: e.g., "Lavender", "Empty", "Noise source"
   - (Repeat for Beaker 2 and 3)

   **Experimental Conditions:**
   - ☑ Faraday Cage Used (check if yes)
   - Experiment ID: Select or type new (e.g., "EXP_2026_001")
   - Researcher Name: Your name
   - Description: What you're testing

7. **Fetch Weather Data:**
   - Click **"Fetch Current Weather (Viroflay)"**
   - Confirms weather capture

8. **Save Metadata:**
   - **"Save Metadata"** - Saves but keeps as draft
   - **"Save & Mark Complete"** - Saves and marks ready for processing

---

### Method 2: Annotate Existing Recordings

If you've already recorded on rubix44 manually:

1. **Open:** http://localhost:5001/annotate

2. **Refresh list** if needed (button in "Available Recordings")

3. **Click on recording** in left panel

4. **Follow steps 6-8 above** to fill and save metadata

---

## 🔄 Automatic Processing

Once a recording is **marked complete** and **approved**, the system will:

1. **Rubix44 provider polls** (every 5 minutes)
2. **Checks MariaDB** for:
   - `metadata_complete = TRUE`
   - `quality_approved = TRUE`
3. **Downloads stereo WAV** file
4. **Processes channels** using labels from your metadata
5. **Stores features** in database
6. **Training pipeline** uses features automatically

---

## 📊 Dashboard Overview

**Navigate to:** http://localhost:5001

### Quick Stats
- **Total Recordings**: All sessions in database
- **Pending Metadata**: Need annotation
- **Pending QC**: Metadata complete, awaiting quality check
- **QC Approved**: Ready for/already processed

### Pipeline Status
Real-time status of:
- Rubix44 poller
- Feature extraction
- Model training
- Web interface

### Weather Widget
Current conditions for Viroflay, France (auto-updates)

---

## 🎯 Example: Complete Workflow

### Scenario: Testing Lavender Detection

```bash
# 1. Start web interface
python web/app.py
```

**In browser (http://localhost:5001/annotate):**

```
2. Start New Recording
   - Playback File: noise_baseline.wav
   - Duration: 120 (2 minutes)
   - Output Prefix: lavender_test
   - Click "Start Recording"

3. Wait ~2 minutes for completion

4. Click on new recording: lavender_test_2026-01-04_16-30-00

5. Fill Metadata Form:

   Channel Configuration:
   - Channel 1: Beaker_A
   - Ch1 Class: 1 (Positive - Lavender)
   - Channel 2: Beaker_B
   - Ch2 Class: 0 (Negative - Empty)

   Beaker Setup:
   - Beaker 1: Recording | Lavender
   - Beaker 2: Instrument | Empty
   - Beaker 3: Not Used | (blank)

   Experimental Conditions:
   - ☑ Faraday Cage Used
   - Experiment ID: EXP_2026_001
   - Researcher: Bernd
   - Description: Baseline lavender detection test with noise playback

6. Click "Fetch Current Weather (Viroflay)"

7. Comments: "Good quality, no interference observed"

8. Click "Save & Mark Complete"
```

**System automatically:**
- Saves to MariaDB with `metadata_complete = TRUE`
- Within 5 minutes, rubix44 provider downloads WAV
- Processes stereo channels with correct labels
- Stores 120 seconds × features in database
- Available for training

---

## 🔍 Verification

### Check Database

```bash
# Connect to MariaDB
mysql -h 10.0.0.103 -u devuser -p xor_project

# Check recordings
SELECT session_id, metadata_complete, quality_approved,
       channel_1_source, channel_2_source
FROM recording_sessions
ORDER BY recording_date DESC LIMIT 5;

# Check weather data
SELECT session_id, weather_temperature_c, weather_conditions,
       weather_timestamp
FROM recording_sessions
WHERE weather_temperature_c IS NOT NULL;

# Exit
quit
```

### Check Features Extracted

```bash
mysql -h 10.0.0.103 -u devuser -p xor_project

# Count features from specific recording
SELECT COUNT(*) as feature_count, label
FROM features
WHERE source_file LIKE '%lavender_test%'
GROUP BY label;
```

---

## 🛠️ Troubleshooting

### Recording doesn't appear in list

**Check rubix44 is running:**
```bash
curl http://10.0.0.58:5000/api/v1/recordings/status
```

**Refresh the list:**
- Click "Refresh" button in Available Recordings panel

### Can't start recording

**Check playback files available:**
```bash
curl http://10.0.0.58:5000/api/v1/playback-files
```

**Verify no recording in progress:**
- Status shown in "Start New Recording" card

### Weather fetch fails

**Without API key:**
- System automatically falls back to Open-Meteo (free)
- Should still work

**Check weather service:**
```bash
python src/continuous/weather_service.py
```

### Metadata not saving

**Check database connection:**
```bash
python scripts/init_recording_schema.py
```

**Check logs:**
- Web server console shows errors
- Look for database connection errors

### Processing not automatic

**Verify metadata is complete:**
```sql
SELECT session_id, metadata_complete, quality_approved
FROM recording_sessions
WHERE session_id = 'your-session-id';
```

**Both must be TRUE** for automatic processing

**Check continuous learning orchestrator is running:**
```bash
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/features.db \
    --model-dir models/continuous \
    --report-dir reports/continuous
```

---

## 📋 Metadata Fields Explained

### Required Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `channel_1_source` | What's connected to left channel | "Beaker_A", "Lavender" |
| `channel_1_expected_class` | Ground truth label for channel 1 | 1 (positive) or 0 (negative) |
| `channel_2_source` | What's connected to right channel | "Beaker_B", "Empty" |
| `channel_2_expected_class` | Ground truth label for channel 2 | 0 or 1 |
| `experiment_id` | Experiment identifier | "EXP_2026_001" |
| `researcher_name` | Who performed the recording | "Bernd" |

### Optional but Recommended

| Field | Purpose | Example |
|-------|---------|---------|
| `beaker_X_role` | Beaker function in setup | recording, instrument, empty |
| `beaker_X_content` | What's in the beaker | "Lavender", "Noise source" |
| `faraday_cage_used` | Electromagnetic shielding | true/false |
| `experiment_description` | What you're testing | "Baseline detection test" |
| `comments` | Observations, issues | "Good quality, clean signal" |

### Auto-Captured

| Field | Source | Example |
|-------|--------|---------|
| `weather_temperature_c` | OpenWeatherMap/Open-Meteo | 12.5 |
| `weather_humidity_percent` | Weather API | 72.0 |
| `weather_conditions` | Weather API | "Partly Cloudy" |
| `weather_pressure_hpa` | Weather API | 1013.2 |
| `weather_wind_speed_kmh` | Weather API | 15.3 |
| `recording_date` | Rubix44 session timestamp | 2026-01-04 16:30:00 |
| `duration_seconds` | Rubix44 recording length | 120 |

---

## 🎓 Best Practices

### 1. Naming Conventions

**Experiment IDs:**
- Use consistent format: `EXP_YYYY_NNN`
- Example: `EXP_2026_001`, `EXP_2026_002`

**Output Prefixes:**
- Descriptive: `lavender_test`, `baseline_noise`, `calibration_run`
- Avoid: `test`, `recording`, `tmp`

### 2. Channel Assignment

**Be Consistent:**
- Always use same channel for positive class
- Document your convention
- Example: "Channel 1 = substance, Channel 2 = empty baseline"

### 3. Metadata Quality

**Always fill:**
- Channel sources and expected classes
- Experiment ID and description
- Faraday cage status
- Weather data (click "Fetch")

**Be specific in comments:**
- "Clean recording, no anomalies"
- "Some electrical noise between 0:30-1:00"
- "Excellent class separation observed"

### 4. Weather Capture

**Fetch weather immediately** after recording:
- Weather conditions change
- Captures conditions during actual recording
- Important for environmental correlation analysis

### 5. Testing Before Production

**Test recordings:**
- Short duration (30-60 seconds)
- Verify all equipment working
- Check signal quality
- Then do full production recordings

---

## 📞 Quick Reference

### URLs
- **Dashboard**: http://localhost:5001
- **Annotate**: http://localhost:5001/annotate
- **Rubix44 API**: http://10.0.0.58:5000

### Key Commands

```bash
# Start web interface
python web/app.py

# Check database
mysql -h 10.0.0.103 -u devuser -p xor_project

# Test weather
python src/continuous/weather_service.py

# Test rubix44 connection
curl http://10.0.0.58:5000/api/v1/recordings/status
```

### Status Indicators

- 🟡 **Pending Metadata**: Need to annotate
- 🔵 **Pending QC**: Metadata complete, awaiting approval
- 🟢 **Approved**: Ready for/already processed
- 🔴 **Rejected**: QC failed, don't process

---

## ✅ You're Ready!

You can now:
1. ✅ Start recordings from web interface
2. ✅ Annotate with complete metadata
3. ✅ Auto-capture weather data
4. ✅ Have recordings automatically processed
5. ✅ Monitor pipeline status
6. ✅ Build continuous learning datasets

**Next Steps:**
- Review [WEB_INTERFACE_GUIDE.md](docs/WEB_INTERFACE_GUIDE.md) for advanced features
- Set up quality control visualizations (Phase 2)
- Configure email alerts for pipeline monitoring

Happy recording! 🎙️🔬
