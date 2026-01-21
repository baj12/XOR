

# Recording Management Web Interface Guide

Complete guide for the Recording Management System web interface - manage recording sessions, metadata annotation, quality control, and pipeline monitoring.

## 🚀 Quick Start

### 1. Initialize Database Schema

First, set up the MariaDB tables for recording management:

```bash
# Activate conda environment
source /Users/bernd/miniconda3/bin/activate xorProject

# Initialize schema
python scripts/init_recording_schema.py
```

Expected output:
```
✓ All recording management tables created successfully
✓ Schema verification completed
```

### 2. Set Up Weather API (Optional)

For automatic weather data capture, get a free OpenWeatherMap API key:

1. Sign up at https://openweathermap.org/api
2. Get your API key from the dashboard
3. Set environment variable:

```bash
export OPENWEATHER_API_KEY="your-api-key-here"
```

**Note**: If no API key is provided, the system will use Open-Meteo (free, no key required) as fallback.

### 3. Start the Web Interface

```bash
# Default (runs on port 5001)
python web/app.py

# Custom port and host
python web/app.py --port 8080 --host 0.0.0.0

# Debug mode (for development)
python web/app.py --debug
```

Access the interface at: **http://localhost:5001**

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Web Interface (Flask)                  │
│                    Port: 5001                           │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────┐ ┌─────────────────┐
│   MariaDB    │ │ Rubix44  │ │ Weather Service │
│  (metadata)  │ │   API    │ │  (OpenWeather)  │
└──────────────┘ └──────────┘ └─────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Continuous Learning Pipeline       │
│  - Rubix44DataProvider             │
│  - StereoChannelProcessor          │
│  - FeatureDatabase                 │
│  - IncrementalTrainer              │
└─────────────────────────────────────┘
```

---

## 🗄️ Database Schema

### Key Tables

#### `recording_sessions`
Stores all recording metadata, experimental conditions, and weather data.

**Critical Fields:**
- `session_id`: Unique identifier from rubix44
- `channel_1_expected_class`, `channel_2_expected_class`: Ground truth labels
- `beaker_1/2/3_role` & `_content`: Beaker configuration
- `faraday_cage_used`: Experimental condition
- `weather_*`: Auto-captured weather data
- `metadata_complete`: Ready for processing?
- `quality_approved`: QC status (NULL=pending, TRUE=approved, FALSE=rejected)

#### `qc_visualizations`
Stores UMAP/t-SNE/PCA visualization results for quality control.

#### `pipeline_status`
Real-time status of pipeline components.

#### `experiments`
Experiment registry for organizing recording sessions.

---

## 🌐 Web Interface Pages

### 1. Dashboard (`/`)

**Overview of entire system:**
- Quick stats (total recordings, pending metadata/QC, approved)
- Pipeline status (real-time component monitoring)
- Recent recordings list
- Current weather widget (Viroflay, France)
- Quick actions menu

**Auto-refreshes every 30 seconds**

### 2. Recordings List (`/recordings`)
*(To be implemented in next phase)*

- View all recordings with filtering
- Filter by: experiment, date range, status (metadata/QC)
- Bulk actions: approve, reject, delete
- Export to CSV

### 3. Annotation Interface (`/annotate`)
*(To be implemented in next phase)*

**Purpose:** Add metadata to new recordings from Rubix44

**Workflow:**
1. Fetch available recordings from Rubix44 API
2. Select recording to annotate
3. Fill form:
   - Channel configuration (what's connected to each channel)
   - Expected class labels (0 or 1)
   - Beaker setup (3 beakers: role and content)
   - Faraday cage status
   - Experiment ID and description
   - Researcher name and comments
4. Auto-fetch weather data for Viroflay
5. Save to MariaDB

**Form Fields:**
```
Session ID: [auto-filled from rubix44]
Recording Date: [auto-filled]

┌─ Channel Configuration ────────────────┐
│ Channel 1 (Left):  [Dropdown: Beaker_A/B/C, Instrument, etc.]
│ Expected Class:    [0 - Negative] [1 - Positive]
│
│ Channel 2 (Right): [Dropdown: Beaker_A/B/C, Instrument, etc.]
│ Expected Class:    [0 - Negative] [1 - Positive]
└────────────────────────────────────────┘

┌─ Beaker Setup ─────────────────────────┐
│ Beaker 1:
│   Role: [Recording/Instrument/Empty/Not Used]
│   Content: [Text: e.g., "Lavender", "Empty", "Noise source"]
│
│ Beaker 2: (same as above)
│ Beaker 3: (same as above)
└────────────────────────────────────────┘

Faraday Cage: [✓] Used  [ ] Not Used

Experiment ID: [Dropdown or text: EXP_2026_001]
Researcher: [Text: Bernd]

Description: [Textarea]
Comments: [Textarea]

[Fetch Weather] → Auto-fills weather data

[Save Metadata] [Cancel]
```

### 4. Quality Control (`/qc`)
*(To be implemented in next phase)*

**Purpose:** Review recordings with visualizations before training

**Features:**
- Generate UMAP/t-SNE/PCA plots (2D, 3D, 4D views)
- Interactive scatter plots (Plotly.js)
- Class separation metrics
- Approve/Reject/Needs Review decisions
- QC notes

**Visualization Tabs:**
- UMAP 2D
- t-SNE 2D
- PCA Dimensions 1-2
- PCA Dimensions 3-4

### 5. Pipeline Monitor (`/pipeline`)
*(To be implemented in next phase)*

**Real-time pipeline monitoring:**
- Component status (Rubix44 poller, feature extraction, training)
- Progress bars for active tasks
- Error logs and alerts
- Database statistics
- Resource usage (CPU, memory, GPU)

### 6. Training Monitor (`/training`)
*(To be implemented in next phase)*

**Live training metrics:**
- Accuracy/Loss curves over time
- Confusion matrix
- Feature importance
- Drift detection alerts
- Performance history charts

---

## 📡 API Endpoints

### Recordings

#### `GET /api/recordings`
Get list of recordings with filtering.

**Query Parameters:**
- `status`: `pending_metadata`, `pending_qc`, `approved`, `rejected`
- `experiment_id`: Filter by experiment
- `limit`: Max results (default: 100)
- `offset`: Pagination offset (default: 0)

**Response:**
```json
{
  "success": true,
  "recordings": [
    {
      "session_id": "recording_2026-01-04_15-30-00",
      "recording_date": "2026-01-04T15:30:00",
      "channel_1_source": "Beaker_A",
      "channel_2_source": "Beaker_B",
      "metadata_complete": true,
      "quality_approved": true,
      ...
    }
  ],
  "total": 127,
  "limit": 100,
  "offset": 0
}
```

#### `GET /api/recordings/<session_id>`
Get single recording details.

#### `POST /api/recordings/<session_id>`
Create or update recording metadata.

**Request Body:**
```json
{
  "channel_1_source": "Beaker_A",
  "channel_2_source": "Beaker_B",
  "channel_1_expected_class": 1,
  "channel_2_expected_class": 0,
  "beaker_1_role": "recording",
  "beaker_1_content": "Lavender",
  "beaker_2_role": "instrument",
  "beaker_2_content": "Empty",
  "beaker_3_role": "not_used",
  "faraday_cage_used": true,
  "experiment_id": "EXP_2026_001",
  "researcher_name": "Bernd",
  "experiment_description": "Testing lavender detection",
  "comments": "Good quality recording",
  "metadata_complete": true
}
```

### Weather

#### `GET /api/weather`
Get current weather for Viroflay, France.

**Response:**
```json
{
  "success": true,
  "weather": {
    "weather_temperature_c": 12.5,
    "weather_humidity_percent": 72.0,
    "weather_pressure_hpa": 1013.2,
    "weather_conditions": "Partly Cloudy",
    "weather_wind_speed_kmh": 15.3,
    "weather_wind_direction": "NW",
    "weather_timestamp": "2026-01-04T15:30:00",
    "weather_api_source": "OpenWeatherMap"
  },
  "summary": "12.5°C, 72% humidity, Partly Cloudy"
}
```

### Rubix44 Integration

#### `GET /api/rubix44/recordings`
Get available recordings from Rubix44 API.

**Response:**
```json
{
  "success": true,
  "recordings": [
    {
      "id": "recording_2026-01-04_15-30-00",
      "prefix": "recording",
      "timestamp": "2026-01-04_15-30-00",
      "files": [
        {
          "name": "recording_2026-01-04_15-30-00_stereo.wav",
          "size": 1764044,
          "path": "recordings\\recording_2026-01-04_15-30-00_stereo.wav"
        }
      ]
    }
  ],
  "api_url": "http://10.0.0.58:5000"
}
```

### Pipeline

#### `GET /api/pipeline/status`
Get current status of all pipeline components.

**Response:**
```json
{
  "success": true,
  "statuses": [
    {
      "component": "rubix44_poller",
      "status": "running",
      "message": "Polling every 5 minutes",
      "timestamp": "2026-01-04T15:30:00",
      "metrics": {}
    },
    {
      "component": "feature_extraction",
      "status": "idle",
      "message": "Waiting for approved recordings",
      "timestamp": "2026-01-04T15:30:00"
    }
  ]
}
```

### Experiments

#### `GET /api/experiments`
Get list of experiments.

---

## 🔄 Complete Workflow

### Recording → Training Pipeline

```
1. Recording Made on Rubix44
   └─> Creates stereo WAV file
   └─> Appears in rubix44 API history

2. Web Interface: Annotate
   └─> Fetch from Rubix44 API
   └─> Fill metadata form
   └─> Auto-fetch weather
   └─> Save to MariaDB (metadata_complete = TRUE)

3. Web Interface: Quality Control
   └─> Load recording features
   └─> Generate UMAP/t-SNE/PCA visualizations
   └─> Review class separation
   └─> Approve/Reject (quality_approved = TRUE/FALSE)

4. Rubix44 Data Provider (Automatic)
   └─> Polls API every 5 minutes
   └─> Queries MariaDB for metadata
   └─> Only processes if:
       - metadata_complete = TRUE
       - quality_approved = TRUE
   └─> Downloads stereo WAV
   └─> Processes with StereoChannelProcessor
   └─> Stores features with correct labels from metadata

5. Continuous Learning Pipeline
   └─> Incremental training on new features
   └─> Drift detection
   └─> Performance monitoring
   └─> Weekly reports
```

---

## 🛠️ Configuration

### Environment Variables

```bash
# Weather API (optional, falls back to Open-Meteo)
export OPENWEATHER_API_KEY="your-key-here"

# Rubix44 API URL
export RUBIX44_URL="http://10.0.0.58:5000"

# Flask secret key (production)
export FLASK_SECRET_KEY="your-secret-key"

# MariaDB credentials (from ../.env)
MARIADB_HOST=10.0.0.103
MARIADBUSER=devuser
MARIADBDEVPWD=<password>
MARIADB_DATABASE=xor_project
```

### continuous_learning_config.yaml

Key settings for integration:

```yaml
orchestration:
  data_provider: rubix44  # Use rubix44 API

  rubix44:
    api_url: http://10.0.0.58:5000
    poll_interval_minutes: 5
    download_dir: data/continuous/recordings
    cleanup_after_processing: false  # Keep WAVs for review
```

---

## 📝 CSV Batch Import

*(To be implemented)*

Import multiple recordings from CSV file:

```bash
python scripts/import_metadata_csv.py \
    --csv recordings_metadata.csv \
    --fetch-weather \
    --auto-approve
```

**CSV Format:**
```csv
session_id,recording_date,channel_1_source,channel_2_source,ch1_class,ch2_class,beaker_1_role,beaker_1_content,beaker_2_role,beaker_2_content,beaker_3_role,beaker_3_content,faraday_cage,experiment_id,researcher,description,comments
recording_2026-01-04_15-30-00,2026-01-04 15:30:00,Beaker_A,Beaker_B,1,0,recording,Lavender,instrument,Empty,not_used,,true,EXP_2026_001,Bernd,Testing lavender detection,Good quality
```

---

## 🧪 Testing

### Test Weather Service

```bash
python src/continuous/weather_service.py
```

### Test Database Connection

```bash
python scripts/init_recording_schema.py
```

### Test Web Interface

```bash
# Start web server
python web/app.py --debug

# In another terminal, test API
curl http://localhost:5001/api/weather
curl http://localhost:5001/api/recordings
curl http://localhost:5001/api/pipeline/status
```

---

## 🚨 Troubleshooting

### Weather API Not Working

- Check API key: `echo $OPENWEATHER_API_KEY`
- Fallback to Open-Meteo (no key required) should work automatically
- Check logs for specific error messages

### Database Connection Errors

- Verify MariaDB is running: `mysql -h 10.0.0.103 -u devuser -p`
- Check credentials in `../.env`
- Run schema initialization: `python scripts/init_recording_schema.py`

### Rubix44 API Not Accessible

- Check API is running: `curl http://10.0.0.58:5000/api/v1/recordings/status`
- Verify network connectivity to 10.0.0.58
- Check firewall settings

### Web Interface Not Starting

- Check port is not in use: `lsof -i :5001`
- Verify conda environment is activated
- Check Python dependencies: `pip install flask`

---

## 📦 Dependencies

The web interface requires:

```bash
# Python packages
flask>=2.3.0
requests>=2.28.0
plotly>=5.14.0

# Already in environment
mariadb>=1.1.0  # For database
librosa>=0.10.0  # For audio processing
scikit-learn>=1.2.0  # For UMAP/t-SNE/PCA
```

---

## 🔐 Security Notes

### Production Deployment

1. **Change Flask secret key**:
   ```bash
   export FLASK_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   ```

2. **Use HTTPS**: Deploy behind nginx/Apache with SSL

3. **Authentication**: Add login system (not currently implemented)

4. **Database**: Restrict MariaDB access by IP

5. **API Keys**: Never commit API keys to git

---

## 📈 Next Steps

### Phase 2 Features (To Implement)

1. **Annotation Interface** - Complete web form for metadata entry
2. **QC Visualizations** - UMAP/t-SNE/PCA plot generation
3. **CSV Import Tool** - Batch metadata import
4. **Pipeline Monitoring** - Real-time status dashboard
5. **Training Monitor** - Live training metrics
6. **User Authentication** - Login system for multi-user access

### Phase 3 Enhancements

1. **Automated QC** - ML-based quality scoring
2. **Experiment Management** - Create/edit experiments in web UI
3. **Report Generation** - Automated weekly reports
4. **Mobile Interface** - Responsive design for tablets/phones
5. **REST API Documentation** - OpenAPI/Swagger docs

---

## 📞 Support

For issues or questions:
1. Check logs in web server console
2. Review database logs: `mysql -h 10.0.0.103 -u devuser -p xor_project`
3. Test individual components (weather, rubix44, database)
4. Check system documentation in `docs/`

