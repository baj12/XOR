# 24-Hour Continuous Learning Test Run

**Start Time:** $(date)

## Overview

This test run will execute the continuous learning system for approximately 24 hours to validate:

1. **Rubix44 API Integration**: Automatic polling and downloading of stereo recordings
2. **Feature Extraction**: Processing of stereo channels (left=class_0, right=class_1)
3. **Database Storage**: SQLite storage of extracted features
4. **Incremental Training**: Scheduled training based on available data
5. **Monitoring**: System health checks and drift detection
6. **Stability**: 24/7 autonomous operation capability

## Configuration

- **Database**: `data/continuous/test_24hr.db`
- **Models**: `models/continuous_test_24hr/`
- **Reports**: `reports/continuous_test_24hr/`
- **Logs**: `logs/continuous_test_24hr/`

### Settings (from continuous_learning_config.yaml)

- **Data Provider**: rubix44 (API: http://10.0.0.58:5000)
- **Poll Interval**: 5 minutes
- **Training Day**: Monday (day 0)
- **Training Hour**: 2 AM
- **Minimum Samples for Training**: 1000
- **Training Window**: 4 weeks
- **Update Mode**: hybrid (incremental + full retrain every 4 weeks)

## How to Run

### Start the test:
```bash
./run_24hr_test.sh
```

### Monitor in another terminal:
```bash
# Check overall status
./monitor_test_status.sh

# Watch logs in real-time
tail -f logs/continuous_test_24hr/orchestrator_*.log

# Check database stats
sqlite3 data/continuous/test_24hr.db "SELECT COUNT(*) FROM features;"
sqlite3 data/continuous/test_24hr.db "SELECT label, COUNT(*) FROM features GROUP BY label;"
```

### Background execution (recommended for 24-hour test):
```bash
nohup ./run_24hr_test.sh > logs/continuous_test_24hr/nohup.out 2>&1 &
echo $! > logs/continuous_test_24hr/orchestrator.pid
```

### Stop the test:
```bash
# If running in foreground: Ctrl+C

# If running in background:
kill $(cat logs/continuous_test_24hr/orchestrator.pid)
```

## What to Expect

### Initial Phase (First Hour)
- Orchestrator starts and validates rubix44 API connection
- Polls for existing recordings from history
- Downloads and processes any available stereo WAV files
- Extracts features and stores in database
- Initial data collection

### Continuous Operation (Hours 2-23)
- Every 5 minutes: Poll API for new recordings
- If new recordings found: Download → Process → Store features
- Every check: Run monitoring (health checks, drift detection)
- If enough data collected: Training may trigger (if Monday at 2 AM)
- System automatically recovers from transient errors

### Completion (Hour 24)
- Orchestrator stops automatically after 24 hours
- Final status summary in log
- Database contains all processed features
- Models saved (if training occurred)
- Reports generated

## Expected Recordings

Based on API history, the following recordings are available:
- `test_time_estimation_2026-01-12_18-31-04` (60 seconds)
- `test_recording_2026-01-04_18-11-06` (30 seconds)
- `test_orchestrator_2026-01-17_16-19-51` (10 seconds)
- `test_manual_2026-01-15_10-41-57` (180 seconds)

These will be processed during the initial cycle.

## Monitoring Commands

```bash
# API health
curl http://10.0.0.58:5000/api/v1/health

# Recording status
curl http://10.0.0.58:5000/api/v1/recordings/status

# Database sample count
sqlite3 data/continuous/test_24hr.db "SELECT COUNT(*) FROM features;"

# Database class distribution
sqlite3 data/continuous/test_24hr.db "
SELECT
    label,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM features), 2) as percentage
FROM features
GROUP BY label;
"

# Check orchestrator process
ps aux | grep "continuous.orchestrator"

# Disk space
df -h .
```

## Success Criteria

- ✓ Orchestrator runs for full 24 hours without crashing
- ✓ All available recordings are successfully downloaded and processed
- ✓ Features are stored in database with correct schema
- ✓ Monitoring checks execute without errors
- ✓ System handles API unavailability gracefully
- ✓ Error recovery works for transient failures
- ✓ Memory usage remains stable over time
- ✓ No data corruption or loss

## Troubleshooting

### Orchestrator not running
```bash
# Check for errors in log
tail -50 logs/continuous_test_24hr/orchestrator_*.log

# Verify conda environment
conda activate xorProject
python -c "import src.continuous.orchestrator; print('OK')"
```

### No data being ingested
```bash
# Check API connectivity
curl http://10.0.0.58:5000/api/v1/health

# Verify recordings are available
curl http://10.0.0.58:5000/api/v1/recordings/history | python3 -m json.tool

# Check state file
cat data/continuous/recordings/.rubix44_state.json
```

### Database errors
```bash
# Verify database exists and is accessible
sqlite3 data/continuous/test_24hr.db ".schema features"

# Check for locks
lsof data/continuous/test_24hr.db
```

## Post-Test Analysis

After 24 hours, analyze results:

```bash
# Database statistics
sqlite3 data/continuous/test_24hr.db "
SELECT
    'Total Samples' as metric, COUNT(*) as value FROM features
UNION ALL
SELECT 'Class 0', COUNT(*) FROM features WHERE label = 0
UNION ALL
SELECT 'Class 1', COUNT(*) FROM features WHERE label = 1
UNION ALL
SELECT 'Unique Sessions', COUNT(DISTINCT recording_id) FROM features;
"

# Check for any trained models
ls -lh models/continuous_test_24hr/

# Review reports
ls -lh reports/continuous_test_24hr/

# Extract error summary from logs
grep -i "error\|exception\|failed" logs/continuous_test_24hr/*.log | tail -50
```

## Next Steps

Based on test results:
1. Review any errors or warnings in logs
2. Verify feature extraction quality
3. Check class balance in collected data
4. Assess system stability over 24 hours
5. Adjust configuration if needed
6. Plan for longer-term production deployment
