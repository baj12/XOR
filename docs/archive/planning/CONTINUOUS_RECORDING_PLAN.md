# Continuous Recording System - Implementation Plan

**Date**: 2026-01-04
**Status**: Planning Phase
**Goal**: Transform the web-based recording system into a continuous learning application that runs for weeks/months autonomously

---

## Overview

The system will run continuous recording cycles:
1. **Record** for 1 hour (configurable)
2. **QC Check** - Validate recording quality
3. **Train Network** - Incremental learning with new data
4. **Track History** - Log performance metrics
5. **Repeat** - Continue until target duration (weeks/months)

## Architecture Integration

### Existing Infrastructure to Leverage

**Already Implemented:**
- ✅ Web interface for recording control ([web/templates/annotate.html](web/templates/annotate.html))
- ✅ MariaDB metadata storage with QC workflow
- ✅ Rubix44 API integration for recording
- ✅ QC visualization system (UMAP/t-SNE/PCA)
- ✅ Continuous learning modules in [src/continuous/](src/continuous/)
  - `stereo_channel_processor.py` - Audio feature extraction
  - `feature_database.py` - SQLite/MariaDB storage
  - `incremental_trainer.py` - Sliding window training
  - `monitoring_reporter.py` - Health monitoring

**New Components Needed:**
- ⭐ Continuous Recording Orchestrator
- ⭐ Auto-QC validation system
- ⭐ Recording cycle scheduler
- ⭐ Progress dashboard
- ⭐ Experiment configuration manager

---

## System Design

### 1. Continuous Recording Orchestrator

**File**: `src/continuous/recording_orchestrator.py`

**Responsibilities:**
- Manage recording cycles (start → QC → train → repeat)
- Schedule next recording automatically
- Handle failures and retries
- Track experiment progress
- Coordinate with rubix44 API
- Integrate with existing continuous learning modules

**Core Loop:**
```python
class ContinuousRecordingOrchestrator:
    def __init__(self, config):
        self.rubix44_client = Rubix44Client()
        self.processor = StereoChannelProcessor()
        self.trainer = IncrementalTrainer()
        self.qc_validator = AutoQCValidator()
        self.db = DatabaseConnection(backend='mariadb')

    async def run_continuous_experiment(self, weeks: int):
        """Run continuous recording for specified weeks"""
        end_time = datetime.now() + timedelta(weeks=weeks)
        cycle_number = 1

        while datetime.now() < end_time:
            try:
                # 1. Start recording
                session_id = await self.start_recording_cycle(cycle_number)

                # 2. Wait for recording to complete
                await self.wait_for_recording(session_id)

                # 3. Auto-QC validation
                qc_result = await self.validate_recording(session_id)

                if qc_result.passed:
                    # 4. Process features
                    await self.process_features(session_id)

                    # 5. Train network
                    await self.train_incremental(session_id)

                    # 6. Track metrics
                    await self.log_cycle_metrics(cycle_number, session_id)
                else:
                    await self.handle_failed_qc(session_id, qc_result)

                cycle_number += 1

            except Exception as e:
                await self.handle_cycle_error(cycle_number, e)
```

---

### 2. Auto-QC Validation System

**File**: `src/continuous/auto_qc_validator.py`

**Purpose**: Automatically validate recording quality without manual intervention

**Validation Criteria:**
```python
class AutoQCValidator:
    def __init__(self, thresholds_config):
        self.min_separation_score = 0.7  # Silhouette score
        self.min_duration = 3500  # seconds (allow 100s tolerance)
        self.max_duration = 3700
        self.min_samples_per_channel = 900  # per channel

    async def validate_recording(self, session_id) -> QCResult:
        """
        Automatic quality validation based on:
        1. Duration check
        2. File integrity check
        3. Feature extraction test
        4. Class separation analysis (UMAP/t-SNE)
        5. Audio quality metrics (SNR, clipping detection)
        """

        # Get recording metadata
        metadata = await self.get_recording_metadata(session_id)

        # Check 1: Duration
        duration_ok = self.check_duration(metadata)

        # Check 2: File exists and is readable
        file_ok = self.check_file_integrity(metadata)

        # Check 3: Extract features (limited sample)
        features_ok, features = self.extract_test_features(metadata)

        # Check 4: Class separation analysis
        separation_ok, separation_score = self.analyze_class_separation(features)

        # Check 5: Audio quality
        audio_ok, audio_metrics = self.check_audio_quality(metadata)

        # Overall decision
        passed = all([duration_ok, file_ok, features_ok, separation_ok, audio_ok])

        return QCResult(
            passed=passed,
            duration_check=duration_ok,
            file_check=file_ok,
            features_check=features_ok,
            separation_score=separation_score,
            audio_metrics=audio_metrics,
            recommendation="approve" if passed else "reject"
        )
```

**Integration with Existing QC:**
- Uses existing `StereoChannelProcessor` for feature extraction
- Generates same UMAP/t-SNE/PCA plots as manual QC page
- Stores results in `qc_visualizations` table
- Can be reviewed manually later if needed

---

### 3. Enhanced Web Interface

#### A. Continuous Experiment Page

**File**: `web/templates/continuous_experiment.html`

**Features:**
- Start new continuous experiment
- Configure duration (weeks/months)
- Set recording interval (default 1 hour)
- Define auto-QC thresholds
- View live progress

**UI Layout:**
```
┌─────────────────────────────────────────────┐
│ Continuous Learning Experiment              │
├─────────────────────────────────────────────┤
│ Configuration                                │
│ ┌─────────────────────────────────────────┐ │
│ │ Experiment Name: [Lavender Study 2026] │ │
│ │ Duration: [4] weeks                     │ │
│ │ Recording Interval: [60] minutes        │ │
│ │ Playback File: [exp6 - 3 noise.wav]    │ │
│ │ Channel 1 (Left): Lavender [Class: 1]  │ │
│ │ Channel 2 (Right): Empty   [Class: 0]  │ │
│ │                                         │ │
│ │ Auto-QC Thresholds:                     │ │
│ │ - Min Separation Score: [0.7]           │ │
│ │ - Min Samples/Channel: [900]            │ │
│ │                                         │ │
│ │ [Start Continuous Experiment]           │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Active Experiments                           │
├─────────────────────────────────────────────┤
│ Lavender Study 2026                          │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45% (Cycle 78) │
│ Started: 2026-01-01 | Est. End: 2026-01-29  │
│ Status: ⏸ Recording (Cycle 79)               │
│ Last QC: ✅ Passed (0.82 separation)         │
│ Model Accuracy: 94.2% (↑ 0.3% from baseline)│
│ [View Details] [Pause] [Stop]                │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Cycle History (Last 10)                      │
├─────────────────────────────────────────────┤
│ Cycle | Time     | QC    | Samples | Acc   │
│ 78    | 14:00    | ✅ 0.81| 1850    | 94.2% │
│ 77    | 13:00    | ✅ 0.79| 1820    | 93.9% │
│ 76    | 12:00    | ❌ 0.65| 1650    | -     │
│ 75    | 11:00    | ✅ 0.83| 1900    | 93.9% │
└─────────────────────────────────────────────┘
```

#### B. Progress Dashboard

**File**: `web/templates/continuous_dashboard.html`

**Real-time Metrics:**
- Current cycle number
- Progress bar (cycles completed / total expected)
- QC pass rate
- Model performance over time (line chart)
- Data collected (total samples)
- Storage usage
- Estimated completion time

**Charts:**
1. **Accuracy Over Time** (line chart)
2. **QC Pass Rate** (bar chart by day)
3. **Class Separation Scores** (scatter plot)
4. **Training Time per Cycle** (histogram)

---

### 4. Database Schema Extensions

#### New Table: `continuous_experiments`

```sql
CREATE TABLE continuous_experiments (
    experiment_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(200) NOT NULL,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    target_duration_weeks INT NOT NULL,
    recording_interval_minutes INT DEFAULT 60,
    playback_file VARCHAR(200),
    channel_1_class INT,
    channel_2_class INT,
    status ENUM('running', 'paused', 'completed', 'failed') DEFAULT 'running',
    current_cycle INT DEFAULT 0,
    total_cycles_expected INT,
    qc_pass_count INT DEFAULT 0,
    qc_fail_count INT DEFAULT 0,
    total_samples_collected INT DEFAULT 0,
    baseline_accuracy FLOAT,
    current_accuracy FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

#### New Table: `recording_cycles`

```sql
CREATE TABLE recording_cycles (
    cycle_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    cycle_number INT NOT NULL,
    session_id VARCHAR(100),  -- Links to recording_sessions
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_seconds INT,
    qc_passed BOOLEAN,
    qc_separation_score FLOAT,
    qc_notes TEXT,
    samples_extracted INT,
    features_processed BOOLEAN DEFAULT FALSE,
    training_completed BOOLEAN DEFAULT FALSE,
    model_accuracy FLOAT,
    training_time_seconds FLOAT,
    status ENUM('recording', 'qc_pending', 'qc_failed', 'training', 'completed', 'failed'),
    error_message TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES continuous_experiments(experiment_id),
    FOREIGN KEY (session_id) REFERENCES recording_sessions(session_id),
    INDEX idx_experiment_cycle (experiment_id, cycle_number)
);
```

#### Extend `recording_sessions` table:

```sql
ALTER TABLE recording_sessions
ADD COLUMN experiment_id VARCHAR(100),
ADD COLUMN cycle_number INT,
ADD COLUMN auto_qc_passed BOOLEAN,
ADD COLUMN auto_qc_score FLOAT,
ADD FOREIGN KEY (experiment_id) REFERENCES continuous_experiments(experiment_id);
```

---

### 5. Configuration Management

**File**: `config/continuous_experiment_config.yaml`

```yaml
experiment:
  name: "Lavender Continuous Study 2026"
  description: "4-week continuous learning experiment with lavender samples"
  duration_weeks: 4
  recording_interval_minutes: 60

recording:
  playback_file: "exp6 - 3 noise.wav"
  duration_seconds: 3600
  output_prefix: "continuous"

channels:
  channel_1:
    source: "Lavender"
    expected_class: 1
  channel_2:
    source: "Empty"
    expected_class: 0

beakers:
  beaker_1:
    role: "instrument"
    content: "Rubix"
  beaker_2:
    role: "recording"
    content: "Lavender"

auto_qc:
  enabled: true
  min_separation_score: 0.7
  min_samples_per_channel: 900
  min_duration_seconds: 3500
  max_duration_seconds: 3700
  auto_approve_threshold: 0.8  # Above this, auto-approve
  auto_reject_threshold: 0.6   # Below this, auto-reject
  manual_review_range: [0.6, 0.8]  # In between, flag for manual review

training:
  incremental: true
  sliding_window_weeks: 2
  batch_size: 32
  epochs_per_cycle: 5
  learning_rate: 0.0001
  validation_split: 0.2
  save_model_every_n_cycles: 10

monitoring:
  email_alerts: true
  alert_on_qc_failure: true
  alert_on_training_degradation: true
  performance_threshold: 0.85  # Alert if accuracy drops below this
  weekly_report: true

storage:
  database_backend: "mariadb"
  keep_raw_audio: true  # Keep WAV files or delete after feature extraction
  compress_features: true
  backup_every_n_cycles: 50
```

---

## Implementation Phases

### Phase 1: Core Orchestrator (Week 1)
**Priority**: High

**Tasks:**
1. Create `ContinuousRecordingOrchestrator` class
2. Implement recording cycle loop
3. Integrate with existing rubix44 API
4. Add error handling and retry logic
5. Create database tables
6. Basic logging

**Deliverables:**
- Working orchestrator that can run multiple recording cycles
- Database tracking of cycles
- Command-line interface to start/stop

**Testing:**
- Run 10-cycle test (10 hours)
- Verify cycle transitions
- Test failure recovery

---

### Phase 2: Auto-QC System (Week 2)
**Priority**: High

**Tasks:**
1. Create `AutoQCValidator` class
2. Integrate with `StereoChannelProcessor`
3. Implement class separation analysis
4. Define quality thresholds
5. Generate QC visualizations automatically
6. Store results in database

**Deliverables:**
- Automated QC that matches manual QC accuracy
- Configurable thresholds
- Automatic approve/reject decisions
- Manual review flagging for edge cases

**Testing:**
- Compare auto-QC with manual QC on 50 recordings
- Tune thresholds for >95% agreement
- Test edge cases (low separation, clipping, etc.)

---

### Phase 3: Incremental Training Integration (Week 2-3)
**Priority**: High

**Tasks:**
1. Integrate with existing `IncrementalTrainer`
2. Implement sliding window training (configurable weeks)
3. Track model performance over time
4. Save model checkpoints
5. Implement rollback on performance degradation
6. Add training metrics to database

**Deliverables:**
- Continuous learning with sliding window
- Performance tracking over cycles
- Model versioning
- Automatic rollback capability

**Testing:**
- Run 50-cycle test
- Verify model improves over time
- Test rollback mechanism

---

### Phase 4: Web Interface (Week 3)
**Priority**: Medium

**Tasks:**
1. Create continuous experiment configuration page
2. Create live progress dashboard
3. Add cycle history view
4. Implement start/pause/stop controls
5. Add performance charts (accuracy over time, QC rates)
6. Create experiment management (list, view, delete)

**Deliverables:**
- Full web UI for continuous experiments
- Real-time progress monitoring
- Historical analysis tools

**Testing:**
- User testing with real experiment
- Verify real-time updates
- Test responsive design

---

### Phase 5: Monitoring & Alerts (Week 4)
**Priority**: Medium

**Tasks:**
1. Integrate with existing `MonitoringReporter`
2. Add email alerts for failures
3. Weekly performance reports
4. Storage monitoring
5. Health checks
6. Automatic experiment summary on completion

**Deliverables:**
- Email alerts for critical events
- Weekly reports
- Completion summary with all metrics

**Testing:**
- Test alert triggers
- Verify report generation
- Run end-to-end 1-week experiment

---

### Phase 6: Production Hardening (Week 5)
**Priority**: Medium

**Tasks:**
1. Add comprehensive error handling
2. Implement graceful shutdown
3. Add experiment pause/resume
4. Create backup/restore functionality
5. Optimize storage (compress/archive old data)
6. Performance optimization
7. Documentation

**Deliverables:**
- Production-ready system
- Complete documentation
- Backup procedures
- Troubleshooting guide

---

## API Endpoints

### New Flask Routes

```python
# Continuous Experiments
@app.route('/api/continuous/experiments', methods=['GET'])
def list_continuous_experiments()

@app.route('/api/continuous/experiments', methods=['POST'])
def create_continuous_experiment()

@app.route('/api/continuous/experiments/<exp_id>', methods=['GET'])
def get_experiment_status()

@app.route('/api/continuous/experiments/<exp_id>/pause', methods=['POST'])
def pause_experiment()

@app.route('/api/continuous/experiments/<exp_id>/resume', methods=['POST'])
def resume_experiment()

@app.route('/api/continuous/experiments/<exp_id>/stop', methods=['POST'])
def stop_experiment()

# Cycle Management
@app.route('/api/continuous/experiments/<exp_id>/cycles', methods=['GET'])
def get_experiment_cycles()

@app.route('/api/continuous/experiments/<exp_id>/current-cycle', methods=['GET'])
def get_current_cycle_status()

# Metrics
@app.route('/api/continuous/experiments/<exp_id>/metrics', methods=['GET'])
def get_experiment_metrics()

@app.route('/api/continuous/experiments/<exp_id>/performance', methods=['GET'])
def get_performance_history()

# Auto-QC
@app.route('/api/continuous/auto-qc/<session_id>', methods=['POST'])
def run_auto_qc()
```

---

## File Structure

```
XOR/
├── src/
│   ├── continuous/
│   │   ├── __init__.py
│   │   ├── stereo_channel_processor.py       # ✅ Exists
│   │   ├── feature_database.py               # ✅ Exists
│   │   ├── incremental_trainer.py            # ✅ Exists
│   │   ├── monitoring_reporter.py            # ✅ Exists
│   │   ├── recording_orchestrator.py         # ⭐ NEW
│   │   ├── auto_qc_validator.py              # ⭐ NEW
│   │   ├── experiment_manager.py             # ⭐ NEW
│   │   └── cycle_scheduler.py                # ⭐ NEW
│   └── ...
├── web/
│   ├── app.py                                # Update with new routes
│   ├── templates/
│   │   ├── base.html                         # ✅ Exists
│   │   ├── index.html                        # ✅ Exists (dashboard)
│   │   ├── annotate.html                     # ✅ Exists
│   │   ├── recordings.html                   # ✅ Exists
│   │   ├── qc.html                           # ✅ Exists
│   │   ├── continuous_experiment.html        # ⭐ NEW
│   │   ├── continuous_dashboard.html         # ⭐ NEW
│   │   └── experiment_history.html           # ⭐ NEW
│   └── static/
│       ├── css/
│       │   └── continuous.css                # ⭐ NEW
│       └── js/
│           └── continuous_dashboard.js       # ⭐ NEW
├── config/
│   └── continuous_experiment_config.yaml     # ⭐ NEW
├── scripts/
│   ├── init_continuous_schema.sql            # ⭐ NEW
│   ├── init_continuous_schema.py             # ⭐ NEW
│   ├── start_continuous_experiment.py        # ⭐ NEW
│   └── stop_continuous_experiment.py         # ⭐ NEW
└── docs/
    ├── CONTINUOUS_RECORDING_PLAN.md          # This file
    ├── CONTINUOUS_EXPERIMENT_GUIDE.md        # ⭐ NEW (user guide)
    └── CONTINUOUS_API_REFERENCE.md           # ⭐ NEW (API docs)
```

---

## Key Technical Decisions

### 1. Recording Schedule

**Option A: Fixed Interval** (Recommended)
- Record every hour on the hour
- Predictable, easy to monitor
- Works well with sliding window training

**Option B: Continuous with Overlap**
- Start next recording immediately after QC/training
- Maximizes data collection
- More complex scheduling

**Decision**: Start with Option A, can add Option B later

---

### 2. QC Failure Handling

**Strategy:**
1. **Auto-retry once** - Maybe transient issue (e.g., network glitch)
2. **Flag for manual review** - Store recording, don't train on it
3. **Alert if multiple consecutive failures** - Something is wrong
4. **Continue experiment** - Don't stop entire run for one bad cycle

**Thresholds:**
- 3 consecutive failures → Email alert
- 10 consecutive failures → Auto-pause experiment

---

### 3. Training Strategy

**Sliding Window** (Recommended):
- Train on last N weeks of data (e.g., 2 weeks)
- Prevents drift from very old data
- Adapts to changing conditions
- Configurable window size

**Full History**:
- Train on all data from experiment start
- Better for stable environments
- Risk of overfitting to early conditions

**Decision**: Use sliding window, make configurable

---

### 4. Storage Management

**Raw Audio Files:**
- Keep for 30 days
- Archive to external storage after 30 days
- Delete after 90 days (keep features only)

**Features:**
- Keep all in database (compressed)
- Relatively small (~1-2 GB for 4 weeks)

**Models:**
- Keep checkpoint every 10 cycles
- Keep all models during experiment
- Archive after completion

---

### 5. Performance Monitoring

**Key Metrics:**
1. **Validation Accuracy** - Primary metric
2. **QC Pass Rate** - Data quality indicator
3. **Class Separation Score** - Feature quality
4. **Training Time** - Resource usage
5. **Storage Growth** - Disk space monitoring

**Alerts:**
- Accuracy drops >5% from baseline
- QC pass rate <70%
- Storage >90% full
- Training time doubles (indicates problem)

---

## Success Criteria

### Phase 1 Success:
- [ ] Can run 24-hour autonomous experiment (24 cycles)
- [ ] All cycles complete without manual intervention
- [ ] Database correctly tracks all cycles
- [ ] Error recovery works (test by killing process mid-cycle)

### Phase 2 Success:
- [ ] Auto-QC agrees with manual QC >95% of time
- [ ] No false positives (rejecting good recordings)
- [ ] <5% false negatives (accepting bad recordings)
- [ ] Thresholds are well-tuned

### Phase 3 Success:
- [ ] Model accuracy improves over 50 cycles
- [ ] Sliding window training works correctly
- [ ] Rollback mechanism tested and working
- [ ] Model checkpoints saved correctly

### Phase 4 Success:
- [ ] Web UI shows real-time progress
- [ ] Can start/pause/stop experiments from UI
- [ ] Charts update every 5 minutes
- [ ] Works on mobile browsers

### Phase 5 Success:
- [ ] Email alerts sent for all critical events
- [ ] Weekly reports generated automatically
- [ ] All monitoring metrics collected
- [ ] Completion summary generated

### Final Success (Production):
- [ ] Run 4-week experiment successfully
- [ ] >90% QC pass rate
- [ ] Model accuracy improves or stable
- [ ] No manual intervention required
- [ ] Complete documentation available

---

## Risk Mitigation

### Risk 1: Rubix44 Server Downtime
**Mitigation:**
- Retry logic with exponential backoff
- Alert on repeated failures
- Auto-pause experiment if >1 hour downtime
- Resume automatically when server returns

### Risk 2: Disk Space
**Mitigation:**
- Monitor storage every cycle
- Auto-archive old recordings
- Alert at 80% full
- Stop new recordings at 95% full

### Risk 3: Model Performance Degradation
**Mitigation:**
- Track baseline accuracy
- Alert on >5% drop
- Automatic rollback to previous model
- Flag for manual review

### Risk 4: Network Issues
**Mitigation:**
- Local caching of rubix44 responses
- Retry with timeout
- Store partial results
- Resume from last successful cycle

### Risk 5: Database Corruption
**Mitigation:**
- Daily backups
- Transaction-based writes
- Foreign key constraints
- Regular integrity checks

---

## Next Steps

### Immediate Actions:
1. **Review this plan** - Get feedback on architecture
2. **Choose start date** - When to begin Phase 1?
3. **Allocate resources** - Server capacity, storage
4. **Set up development environment** - Test database, rubix44 connection

### Week 1 Tasks:
1. Create database schema
2. Implement basic orchestrator
3. Write unit tests
4. Run 10-cycle test

### Questions to Answer:
1. What duration for first production experiment? (2 weeks? 4 weeks?)
2. What QC thresholds should we use initially?
3. How much storage is available?
4. Should we run test experiment first?

---

**This plan transforms the current one-off recording system into a fully autonomous continuous learning platform that can run for weeks without intervention!**

**Estimated Total Implementation Time**: 4-5 weeks
**Complexity**: Medium-High
**Dependencies**: Existing continuous learning modules, rubix44 server
**Impact**: Enables true continuous learning experiments
