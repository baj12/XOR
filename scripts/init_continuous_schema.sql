-- Continuous Learning Experiment Schema
-- Database: xor_project
-- Date: 2026-01-04

USE xor_project;

-- Table: continuous_experiments
-- Tracks long-running continuous learning experiments
CREATE TABLE IF NOT EXISTS continuous_experiments (
    experiment_id VARCHAR(100) PRIMARY KEY,
    experiment_name VARCHAR(200) NOT NULL,
    description TEXT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    target_duration_weeks INT NOT NULL,
    recording_interval_minutes INT DEFAULT 60,

    -- Recording configuration
    playback_file VARCHAR(200),
    output_prefix VARCHAR(100) DEFAULT 'continuous',
    recording_duration_seconds INT DEFAULT 3600,

    -- Channel configuration
    channel_1_source VARCHAR(100),
    channel_1_expected_class INT,
    channel_2_source VARCHAR(100),
    channel_2_expected_class INT,

    -- Beaker configuration
    beaker_1_role VARCHAR(50),
    beaker_1_content VARCHAR(100),
    beaker_2_role VARCHAR(50),
    beaker_2_content VARCHAR(100),

    -- Experimental conditions
    faraday_cage_used BOOLEAN DEFAULT FALSE,
    researcher_name VARCHAR(100) DEFAULT 'Bernd',

    -- Auto-QC configuration
    auto_qc_enabled BOOLEAN DEFAULT TRUE,
    auto_qc_min_separation_score FLOAT DEFAULT 0.7,
    auto_qc_min_samples_per_channel INT DEFAULT 900,
    auto_qc_auto_approve_threshold FLOAT DEFAULT 0.8,
    auto_qc_auto_reject_threshold FLOAT DEFAULT 0.6,

    -- Training configuration
    training_sliding_window_weeks INT DEFAULT 2,
    training_batch_size INT DEFAULT 32,
    training_epochs_per_cycle INT DEFAULT 5,
    training_learning_rate FLOAT DEFAULT 0.0001,

    -- Status tracking
    status ENUM('running', 'paused', 'completed', 'failed', 'stopped') DEFAULT 'running',
    current_cycle INT DEFAULT 0,
    total_cycles_expected INT,
    next_cycle_scheduled DATETIME,

    -- Metrics
    qc_pass_count INT DEFAULT 0,
    qc_fail_count INT DEFAULT 0,
    total_samples_collected INT DEFAULT 0,
    baseline_accuracy FLOAT,
    current_accuracy FLOAT,
    best_accuracy FLOAT,

    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    completed_at DATETIME,

    INDEX idx_status (status),
    INDEX idx_start_time (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Table: recording_cycles
-- Tracks individual recording cycles within an experiment
CREATE TABLE IF NOT EXISTS recording_cycles (
    cycle_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    cycle_number INT NOT NULL,
    session_id VARCHAR(100),  -- Links to recording_sessions

    -- Timing
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration_seconds INT,

    -- QC results
    qc_passed BOOLEAN,
    qc_separation_score FLOAT,
    qc_silhouette_score FLOAT,
    qc_audio_quality_score FLOAT,
    qc_notes TEXT,
    qc_decision ENUM('auto_approved', 'auto_rejected', 'manual_review', 'pending'),

    -- Feature extraction
    samples_extracted_ch1 INT,
    samples_extracted_ch2 INT,
    features_processed BOOLEAN DEFAULT FALSE,
    features_db_imported BOOLEAN DEFAULT FALSE,

    -- Training
    training_completed BOOLEAN DEFAULT FALSE,
    model_accuracy FLOAT,
    model_validation_accuracy FLOAT,
    model_loss FLOAT,
    training_time_seconds FLOAT,
    training_samples_used INT,
    model_checkpoint_path VARCHAR(255),

    -- Status
    status ENUM('scheduled', 'recording', 'qc_pending', 'qc_failed', 'processing', 'training', 'completed', 'failed') DEFAULT 'scheduled',
    error_message TEXT,
    retry_count INT DEFAULT 0,

    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES continuous_experiments(experiment_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES recording_sessions(session_id) ON DELETE SET NULL,

    UNIQUE KEY unique_experiment_cycle (experiment_id, cycle_number),
    INDEX idx_experiment_status (experiment_id, status),
    INDEX idx_start_time (start_time),
    INDEX idx_qc_passed (qc_passed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Extend recording_sessions table with experiment tracking
ALTER TABLE recording_sessions
ADD COLUMN IF NOT EXISTS experiment_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS cycle_number INT,
ADD COLUMN IF NOT EXISTS auto_qc_passed BOOLEAN,
ADD COLUMN IF NOT EXISTS auto_qc_score FLOAT,
ADD COLUMN IF NOT EXISTS auto_qc_decision VARCHAR(50);

-- Add foreign key if not exists (MySQL 8.0+)
SET @fk_exists = (SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = 'xor_project'
    AND TABLE_NAME = 'recording_sessions'
    AND CONSTRAINT_NAME = 'fk_recording_experiment');

SET @sql = IF(@fk_exists = 0,
    'ALTER TABLE recording_sessions ADD CONSTRAINT fk_recording_experiment
     FOREIGN KEY (experiment_id) REFERENCES continuous_experiments(experiment_id) ON DELETE SET NULL',
    'SELECT "Foreign key already exists"');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Create index on experiment_id if not exists
CREATE INDEX IF NOT EXISTS idx_experiment_id ON recording_sessions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_cycle_number ON recording_sessions(cycle_number);

-- Table: experiment_alerts
-- Track alerts and notifications for experiments
CREATE TABLE IF NOT EXISTS experiment_alerts (
    alert_id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL,
    cycle_number INT,
    alert_type ENUM('qc_failure', 'training_degradation', 'disk_space', 'rubix44_error', 'completion', 'paused', 'other') NOT NULL,
    severity ENUM('info', 'warning', 'error', 'critical') DEFAULT 'info',
    message TEXT NOT NULL,
    details JSON,
    email_sent BOOLEAN DEFAULT FALSE,
    email_sent_at DATETIME,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at DATETIME,
    acknowledged_by VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (experiment_id) REFERENCES continuous_experiments(experiment_id) ON DELETE CASCADE,
    INDEX idx_experiment_severity (experiment_id, severity),
    INDEX idx_created_at (created_at),
    INDEX idx_acknowledged (acknowledged)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- View: experiment_summary
-- Quick overview of all experiments
CREATE OR REPLACE VIEW experiment_summary AS
SELECT
    e.experiment_id,
    e.experiment_name,
    e.status,
    e.current_cycle,
    e.total_cycles_expected,
    ROUND(100.0 * e.current_cycle / NULLIF(e.total_cycles_expected, 0), 1) as progress_percent,
    e.qc_pass_count,
    e.qc_fail_count,
    ROUND(100.0 * e.qc_pass_count / NULLIF(e.qc_pass_count + e.qc_fail_count, 0), 1) as qc_pass_rate,
    e.current_accuracy,
    e.baseline_accuracy,
    ROUND((e.current_accuracy - e.baseline_accuracy) * 100, 2) as accuracy_improvement,
    e.total_samples_collected,
    e.start_time,
    e.end_time,
    TIMESTAMPDIFF(HOUR, e.start_time, COALESCE(e.end_time, NOW())) as duration_hours,
    e.next_cycle_scheduled,
    COUNT(DISTINCT a.alert_id) as unacknowledged_alerts
FROM continuous_experiments e
LEFT JOIN experiment_alerts a ON e.experiment_id = a.experiment_id AND a.acknowledged = FALSE
GROUP BY e.experiment_id;

-- View: recent_cycles
-- Last 50 cycles across all experiments with key metrics
CREATE OR REPLACE VIEW recent_cycles AS
SELECT
    rc.cycle_id,
    rc.experiment_id,
    e.experiment_name,
    rc.cycle_number,
    rc.session_id,
    rc.status,
    rc.qc_passed,
    rc.qc_separation_score,
    rc.qc_decision,
    rc.model_accuracy,
    rc.training_time_seconds,
    rc.samples_extracted_ch1 + rc.samples_extracted_ch2 as total_samples,
    rc.start_time,
    rc.end_time,
    TIMESTAMPDIFF(MINUTE, rc.start_time, COALESCE(rc.end_time, NOW())) as duration_minutes
FROM recording_cycles rc
JOIN continuous_experiments e ON rc.experiment_id = e.experiment_id
ORDER BY rc.start_time DESC
LIMIT 50;

-- Initial data: Create default QC thresholds config (stored as JSON in a settings table if needed)
-- For now, thresholds are in continuous_experiments table

-- Grant permissions (adjust user as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON xor_project.continuous_experiments TO 'devuser'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON xor_project.recording_cycles TO 'devuser'@'%';
-- GRANT SELECT, INSERT, UPDATE, DELETE ON xor_project.experiment_alerts TO 'devuser'@'%';

-- Success message
SELECT 'Continuous learning schema created successfully!' as Status;
