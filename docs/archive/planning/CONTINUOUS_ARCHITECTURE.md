# Continuous Recording System - Architecture

**Date**: 2026-01-04

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     WEB INTERFACE (Flask)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ Continuous     │  │ Dashboard      │  │ Experiment      │   │
│  │ Experiment     │  │ (Live Progress)│  │ History         │   │
│  │ Setup Page     │  │                │  │                 │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│         │                    ▲                    ▲             │
└─────────┼────────────────────┼────────────────────┼─────────────┘
          │                    │                    │
          ▼                    │                    │
┌─────────────────────────────────────────────────────────────────┐
│                  CONTINUOUS ORCHESTRATOR                         │
│                 (recording_orchestrator.py)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              MAIN CONTROL LOOP                           │  │
│   │                                                          │  │
│   │  while not experiment_complete:                         │  │
│   │    1. Start Recording  ──────────────────────┐          │  │
│   │    2. Wait for Completion                    │          │  │
│   │    3. Run Auto-QC ──────────────┐            │          │  │
│   │    4. Process Features          │            │          │  │
│   │    5. Train Model ─────────┐    │            │          │  │
│   │    6. Log Metrics          │    │            │          │  │
│   │    7. Sleep until next     │    │            │          │  │
│   │       cycle                │    │            │          │  │
│   └────────────────────────────┼────┼────────────┼──────────┘  │
│                                │    │            │             │
└────────────────────────────────┼────┼────────────┼─────────────┘
                                 │    │            │
            ┌────────────────────┘    │            │
            │    ┌────────────────────┘            │
            │    │    ┌────────────────────────────┘
            ▼    ▼    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     CORE COMPONENTS                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Auto-QC          │  │ Incremental      │  │ Monitoring    │ │
│  │ Validator        │  │ Trainer          │  │ Reporter      │ │
│  │                  │  │                  │  │               │ │
│  │ • Separation     │  │ • Sliding Window │  │ • Email       │ │
│  │   Analysis       │  │ • Model Saving   │  │   Alerts      │ │
│  │ • Audio Quality  │  │ • Performance    │  │ • Weekly      │ │
│  │ • Duration Check │  │   Tracking       │  │   Reports     │ │
│  │ • File Integrity │  │ • Rollback       │  │ • Health      │ │
│  │                  │  │                  │  │   Checks      │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│           │                     │                     │         │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING INFRASTRUCTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │ Stereo Channel   │  │ Feature          │  │ Rubix44       │ │
│  │ Processor        │  │ Database         │  │ Client        │ │
│  │                  │  │                  │  │               │ │
│  │ • MFCC Extract   │  │ • SQLite/MariaDB │  │ • Start/Stop  │ │
│  │ • Spectral       │  │ • Compressed     │  │ • Status      │ │
│  │ • Chroma         │  │   Storage        │  │ • List Files  │ │
│  │ • Tonnetz        │  │ • Query API      │  │               │ │
│  └──────────────────┘  └──────────────────┘  └───────────────┘ │
│           │                     │                     │         │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA STORAGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    MariaDB Database                        │ │
│  │                                                            │ │
│  │  • continuous_experiments  (experiment tracking)          │ │
│  │  • recording_cycles        (cycle details)                │ │
│  │  • recording_sessions      (metadata)                     │ │
│  │  • qc_visualizations       (QC results)                   │ │
│  │  • features                (audio features - compressed)  │ │
│  │  • training_runs           (model performance)            │ │
│  │  • validation_metrics      (accuracy tracking)            │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              File System (Audio & Models)                  │ │
│  │                                                            │ │
│  │  • data/rubix44/recordings/  (WAV files, 30-day retention)│ │
│  │  • models/continuous/        (model checkpoints)          │ │
│  │  • qc_reports/              (visualization plots)         │ │
│  │                                                            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Single Recording Cycle

```
     START CYCLE N
          │
          ▼
    ┌─────────────────────────────────────┐
    │  1. ORCHESTRATOR                    │
    │     • Create recording_cycles entry │
    │     • Status: 'recording'           │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  2. RUBIX44 API                     │
    │     POST /api/v1/recordings/start   │
    │     • playback_file                 │
    │     • duration: 3600s               │
    │     • output_prefix                 │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  3. WAIT FOR COMPLETION             │
    │     Poll /api/v1/recordings/status  │
    │     • Every 30 seconds              │
    │     • Update cycle status           │
    │     • Show progress in UI           │
    └─────────────────────────────────────┘
          │
          ▼ (recording_complete)
    ┌─────────────────────────────────────┐
    │  4. CREATE METADATA                 │
    │     INSERT recording_sessions       │
    │     • session_id                    │
    │     • experiment_id                 │
    │     • cycle_number                  │
    │     • Auto-fill from config:        │
    │       - channel_1/2_expected_class  │
    │       - beaker roles/content        │
    │       - experiment_id               │
    │       - researcher_name             │
    │     • Fetch weather automatically   │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  5. AUTO-QC VALIDATION              │
    │     auto_qc_validator.validate()    │
    │     • Extract test features (1000)  │
    │     • Check duration                │
    │     • Check file integrity          │
    │     • Analyze class separation      │
    │     • Calculate metrics:            │
    │       - Silhouette score            │
    │       - Separation ratio            │
    │       - Audio quality (SNR, clip)   │
    │     • Generate UMAP/t-SNE/PCA plots │
    │     • Decision: PASS/FAIL/REVIEW    │
    └─────────────────────────────────────┘
          │
          ├─ FAIL ──────────────────────────┐
          │                                 ▼
          │                        ┌──────────────────┐
          │                        │ Log failure      │
          │                        │ Send alert       │
          │                        │ SKIP TO NEXT     │
          │                        │ CYCLE            │
          │                        └──────────────────┘
          │
          ▼ PASS
    ┌─────────────────────────────────────┐
    │  6. FEATURE EXTRACTION              │
    │     stereo_channel_processor        │
    │     • Process full WAV file         │
    │     • Extract all samples           │
    │       (~1800 per channel)           │
    │     • Generate features:            │
    │       - MFCC (13 coefficients)      │
    │       - Spectral (7 features)       │
    │       - Chroma (12 bins)            │
    │       - Tonnetz (6 features)        │
    │     • Compress and store            │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  7. INCREMENTAL TRAINING            │
    │     incremental_trainer             │
    │     • Load sliding window data      │
    │       (last 2 weeks by default)     │
    │     • Train model (5 epochs)        │
    │     • Validate on holdout set       │
    │     • Calculate accuracy            │
    │     • Compare to baseline           │
    │     • Save checkpoint               │
    │     • Update validation_metrics     │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  8. LOG METRICS                     │
    │     • Update recording_cycles:      │
    │       - status: 'completed'         │
    │       - model_accuracy              │
    │       - training_time_seconds       │
    │     • Update continuous_experiments:│
    │       - current_cycle += 1          │
    │       - qc_pass_count += 1          │
    │       - current_accuracy            │
    │       - total_samples_collected     │
    └─────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────┐
    │  9. CHECK COMPLETION                │
    │     if current_cycle >= target:     │
    │       • Generate final report       │
    │       • Send completion email       │
    │       • Status: 'completed'         │
    │     else:                           │
    │       • Sleep until next cycle      │
    │       • GO TO START CYCLE N+1       │
    └─────────────────────────────────────┘
```

---

## Timeline Visualization

```
WEEK 1: Setup & Initial Cycles
═══════════════════════════════════════════════════════════════

Hour 0:00  │  START EXPERIMENT
           ▼
Hour 0:00  ├─ Cycle 1  [Recording] ────────────────────────────┐
Hour 1:00  │                                                    │
Hour 1:05  ├─ Cycle 1  [QC] ─────┐                             │
Hour 1:10  │                     │                             │
Hour 1:15  ├─ Cycle 1  [Train] ──┼─────┐                       │
Hour 1:30  │                     │     │                       │
Hour 1:30  ├─ Cycle 1  [Complete]┼─────┼─> Model v1 (baseline) │
           │                     │     │   Accuracy: 87.3%     │
Hour 2:00  ├─ Cycle 2  [Recording]     │                       │
Hour 3:00  │                           │                       │
Hour 3:05  ├─ Cycle 2  [QC] ───────────┘                       │
Hour 3:15  ├─ Cycle 2  [Train] ────────────┐                   │
Hour 3:30  ├─ Cycle 2  [Complete]──────────┼─> Model v2        │
           │                               │   Accuracy: 88.1%  │
           │        ...continues...        │   (+0.8%)          │
           │                               │                    │
Day 7      ├─ Cycle 168 [Complete]────────┴─> Model v168       │
           │                                   Accuracy: 93.2%  │
           │                                   (+5.9% total)    │
           │                                                    │

WEEK 2-4: Continuous Operation
═══════════════════════════════════════════════════════════════

           │  Sliding Window Training Active                   │
           │  (Training on last 2 weeks of data)               │
           │                                                    │
           │  ┌────────────────────────────────────┐           │
           │  │ QC Pass Rate: 94.2%                │           │
           │  │ Average Accuracy: 93.5%            │           │
           │  │ Training Time: ~15 min/cycle       │           │
           │  │ Total Samples: 302,400             │           │
           │  │ Storage Used: 1.8 GB               │           │
           │  └────────────────────────────────────┘           │
           │                                                    │
Week 4     ├─ Cycle 672 [Complete]──────────────> EXPERIMENT   │
           │                                       COMPLETE     │
           ▼                                                    │
     GENERATE FINAL REPORT                                     │
     • Total Cycles: 672                                       │
     • QC Passed: 633 (94.2%)                                  │
     • Final Accuracy: 94.8%                                   │
     • Improvement: +7.5%                                      │
     • Total Samples: 1,139,400                                │
                                                                │
═══════════════════════════════════════════════════════════════
```

---

## Component Interactions

### Orchestrator ↔ Rubix44

```python
# Start Recording
orchestrator.rubix44_client.start_recording(
    playback_file="exp6 - 3 noise.wav",
    duration=3600,
    output_prefix=f"continuous_exp{exp_id}_cycle{cycle_num}"
)

# Poll Status (every 30s)
while True:
    status = orchestrator.rubix44_client.get_status()
    if status['status'] == 'idle':
        break
    await asyncio.sleep(30)

# Get Recording Details
recordings = orchestrator.rubix44_client.list_recordings()
session_id = find_latest_session(recordings)
```

### Orchestrator ↔ Auto-QC

```python
# Run QC Validation
qc_result = await auto_qc_validator.validate_recording(session_id)

if qc_result.passed:
    # Approve and continue
    db.update_recording_session(
        session_id,
        auto_qc_passed=True,
        auto_qc_score=qc_result.separation_score,
        quality_approved=True
    )
else:
    # Reject and skip
    db.update_recording_session(
        session_id,
        auto_qc_passed=False,
        auto_qc_score=qc_result.separation_score,
        quality_approved=False
    )
    await send_qc_failure_alert(session_id, qc_result)
    return  # Skip to next cycle
```

### Orchestrator ↔ Trainer

```python
# Get sliding window data
window_data = feature_db.get_features_since(
    timestamp=datetime.now() - timedelta(weeks=2),
    experiment_id=exp_id
)

# Train incrementally
training_result = await incremental_trainer.train(
    features=window_data,
    epochs=5,
    batch_size=32,
    learning_rate=0.0001
)

# Check for degradation
if training_result.accuracy < baseline_accuracy - 0.05:
    # Rollback to previous model
    await incremental_trainer.rollback()
    await send_degradation_alert(training_result)
else:
    # Save new model
    await incremental_trainer.save_checkpoint(
        f"model_cycle_{cycle_num}.h5"
    )
```

---

## Database Schema Relationships

```
continuous_experiments (1) ──┬──> (N) recording_cycles
                             │
                             └──> (N) recording_sessions
                                       │
                                       ├──> (N) qc_visualizations
                                       │
                                       └──> (N) features


Queries:

1. Get experiment progress:
   SELECT current_cycle, total_cycles_expected,
          qc_pass_count, current_accuracy
   FROM continuous_experiments
   WHERE experiment_id = ?

2. Get recent cycles:
   SELECT cycle_number, qc_passed, model_accuracy, training_time_seconds
   FROM recording_cycles
   WHERE experiment_id = ?
   ORDER BY cycle_number DESC
   LIMIT 10

3. Get performance trend:
   SELECT cycle_number, model_accuracy, qc_separation_score
   FROM recording_cycles
   WHERE experiment_id = ? AND qc_passed = TRUE
   ORDER BY cycle_number

4. Get QC failure rate:
   SELECT
     COUNT(CASE WHEN qc_passed = TRUE THEN 1 END) as passes,
     COUNT(CASE WHEN qc_passed = FALSE THEN 1 END) as fails,
     COUNT(*) as total
   FROM recording_cycles
   WHERE experiment_id = ?
```

---

## Failure Scenarios & Recovery

### Scenario 1: Rubix44 Server Down

```
CYCLE N
  │
  ├─ Start Recording ──> [CONNECTION ERROR]
  │                           │
  │                           ▼
  │                      Retry with backoff
  │                      (5s, 10s, 30s, 60s)
  │                           │
  │                           ├─ SUCCESS ──> Continue
  │                           │
  │                           └─ FAIL after 4 retries
  │                                   │
  │                                   ▼
  │                              Pause experiment
  │                              Send alert
  │                              Wait for manual resume
```

### Scenario 2: QC Failure

```
CYCLE N
  │
  ├─ Auto-QC ──> [FAIL: Separation Score 0.58]
  │                   │
  │                   ├─ Log failure
  │                   ├─ Update cycle status: 'qc_failed'
  │                   ├─ Flag for manual review
  │                   └─ Continue to next cycle
  │
  ├─ CYCLE N+1 (normal operation)
```

### Scenario 3: Training Degradation

```
CYCLE N
  │
  ├─ Train Model ──> [Accuracy: 78.2%, Baseline: 87.3%]
  │                        │
  │                        ▼
  │                   Degradation detected (-9.1%)
  │                        │
  │                        ├─ Rollback to previous model
  │                        ├─ Send alert
  │                        ├─ Flag cycle for investigation
  │                        └─ Continue with old model
  │
  ├─ CYCLE N+1 (with rollback model)
```

### Scenario 4: Disk Full

```
CYCLE N
  │
  ├─ Feature Extraction ──> [DISK FULL ERROR]
  │                              │
  │                              ▼
  │                         Pause experiment
  │                         Send critical alert
  │                         Archive old recordings
  │                         Resume when space available
```

---

This architecture ensures robustness, scalability, and autonomous operation for weeks without manual intervention!
