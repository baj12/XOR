-- Recording Management System Schema for MariaDB
-- This schema supports recording session metadata, weather data, QC visualizations, and pipeline monitoring

USE xor_project;

-- Recording sessions with complete metadata and weather data
CREATE TABLE IF NOT EXISTS recording_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL COMMENT 'Unique session ID from rubix44',
    recording_date DATETIME NOT NULL COMMENT 'Date and time of recording',
    duration_seconds FLOAT COMMENT 'Recording duration in seconds',
    sample_rate INT DEFAULT 44100 COMMENT 'Audio sample rate',

    -- Channel configuration
    channel_1_source VARCHAR(255) COMMENT 'What is connected to channel 1 (left)',
    channel_2_source VARCHAR(255) COMMENT 'What is connected to channel 2 (right)',
    channel_1_expected_class INT COMMENT 'Expected class label for channel 1 (0 or 1)',
    channel_2_expected_class INT COMMENT 'Expected class label for channel 2 (0 or 1)',

    -- Beaker setup (only 3 beakers)
    beaker_1_role ENUM('recording', 'instrument', 'empty', 'not_used') DEFAULT 'not_used',
    beaker_2_role ENUM('recording', 'instrument', 'empty', 'not_used') DEFAULT 'not_used',
    beaker_3_role ENUM('recording', 'instrument', 'empty', 'not_used') DEFAULT 'not_used',
    beaker_1_content VARCHAR(255) COMMENT 'Content description (e.g., Lavender, Empty, Noise)',
    beaker_2_content VARCHAR(255),
    beaker_3_content VARCHAR(255),

    -- Experimental conditions
    faraday_cage_used BOOLEAN DEFAULT FALSE COMMENT 'Was Faraday cage used?',
    experiment_description TEXT COMMENT 'Detailed experiment description',
    experiment_id VARCHAR(100) COMMENT 'Experiment series identifier',
    researcher_name VARCHAR(100) COMMENT 'Name of researcher',

    -- Weather data (captured automatically for Viroflay, France)
    weather_temperature_c FLOAT COMMENT 'Temperature in Celsius',
    weather_humidity_percent FLOAT COMMENT 'Relative humidity percentage',
    weather_pressure_hpa FLOAT COMMENT 'Atmospheric pressure in hPa',
    weather_conditions VARCHAR(255) COMMENT 'Weather conditions description',
    weather_wind_speed_kmh FLOAT COMMENT 'Wind speed in km/h',
    weather_wind_direction VARCHAR(10) COMMENT 'Wind direction (N, NE, E, etc.)',
    weather_timestamp DATETIME COMMENT 'Timestamp of weather observation',
    weather_api_source VARCHAR(50) DEFAULT 'OpenWeatherMap' COMMENT 'Weather API source',

    -- File information
    stereo_filename VARCHAR(500) COMMENT 'Stereo WAV filename',
    ch1_filename VARCHAR(500) COMMENT 'Channel 1 isolated WAV filename',
    ch2_filename VARCHAR(500) COMMENT 'Channel 2 isolated WAV filename',
    file_size_bytes BIGINT COMMENT 'Total file size in bytes',

    -- Annotations
    comments TEXT COMMENT 'Free-form comments and notes',
    tags JSON COMMENT 'Array of tags for searching/filtering',

    -- Processing status flags
    metadata_complete BOOLEAN DEFAULT FALSE COMMENT 'Is metadata fully entered?',
    quality_approved BOOLEAN DEFAULT NULL COMMENT 'QC approval status (NULL=pending, TRUE=approved, FALSE=rejected)',
    imported_to_features_db BOOLEAN DEFAULT FALSE COMMENT 'Features extracted and stored?',
    processed_for_training BOOLEAN DEFAULT FALSE COMMENT 'Used in training?',
    qc_notes TEXT COMMENT 'Quality control notes',

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- Indexes for performance
    INDEX idx_recording_date (recording_date),
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_metadata_complete (metadata_complete),
    INDEX idx_quality_approved (quality_approved),
    INDEX idx_processed (processed_for_training)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Recording session metadata and annotations';

-- QC visualizations (UMAP/t-SNE/PCA results)
CREATE TABLE IF NOT EXISTS qc_visualizations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL COMMENT 'Reference to recording session',
    visualization_type ENUM('umap', 'tsne', 'pca') NOT NULL COMMENT 'Dimensionality reduction method',
    dimensions INT NOT NULL COMMENT 'Number of dimensions (2, 3, or 4)',
    plot_data JSON COMMENT 'Coordinate data for plotting',
    plot_image_path VARCHAR(500) COMMENT 'Path to saved image file',
    class_separation_score FLOAT COMMENT 'Quantitative separation metric',
    variance_explained FLOAT COMMENT 'Variance explained (PCA only)',
    perplexity INT COMMENT 't-SNE perplexity parameter',
    n_neighbors INT COMMENT 'UMAP n_neighbors parameter',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES recording_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session (session_id),
    INDEX idx_viz_type (visualization_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Quality control visualization results';

-- Pipeline monitoring and status
CREATE TABLE IF NOT EXISTS pipeline_status (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL COMMENT 'Status timestamp',
    component VARCHAR(100) NOT NULL COMMENT 'Pipeline component name',
    status ENUM('running', 'idle', 'error', 'completed') NOT NULL COMMENT 'Component status',
    message TEXT COMMENT 'Status message or error description',
    metrics JSON COMMENT 'Component-specific metrics',

    INDEX idx_timestamp (timestamp),
    INDEX idx_component (component),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Pipeline component status tracking';

-- Experiments registry (simplified)
CREATE TABLE IF NOT EXISTS experiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) UNIQUE NOT NULL COMMENT 'Unique experiment identifier',
    experiment_name VARCHAR(255) COMMENT 'Human-readable experiment name',
    description TEXT COMMENT 'Experiment description and objectives',
    start_date DATE COMMENT 'Experiment start date',
    end_date DATE COMMENT 'Experiment end date',
    researcher_name VARCHAR(100) COMMENT 'Primary researcher',
    status ENUM('active', 'completed', 'cancelled') DEFAULT 'active' COMMENT 'Experiment status',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status (status),
    INDEX idx_researcher (researcher_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Experiment registry and tracking';

-- Training runs history (links to existing training_runs table)
-- This view provides a summary of training performance over time
CREATE OR REPLACE VIEW training_summary AS
SELECT
    t.id,
    t.training_date,
    t.model_type,
    t.accuracy,
    t.loss,
    t.precision,
    t.recall,
    t.f1_score,
    COUNT(DISTINCT f.source_file) as num_recordings,
    COUNT(f.id) as num_features
FROM training_runs t
LEFT JOIN features f ON f.timestamp <= t.training_date
GROUP BY t.id
ORDER BY t.training_date DESC;

-- Create a view for recording status dashboard
CREATE OR REPLACE VIEW recording_dashboard AS
SELECT
    r.session_id,
    r.recording_date,
    r.channel_1_source,
    r.channel_2_source,
    r.experiment_id,
    r.researcher_name,
    r.faraday_cage_used,
    r.weather_temperature_c,
    r.weather_conditions,
    r.metadata_complete,
    r.quality_approved,
    r.imported_to_features_db,
    r.processed_for_training,
    COUNT(q.id) as qc_visualizations_count,
    r.created_at,
    r.updated_at
FROM recording_sessions r
LEFT JOIN qc_visualizations q ON r.session_id = q.session_id
GROUP BY r.session_id
ORDER BY r.recording_date DESC;

-- Insert initial pipeline status for monitoring
INSERT INTO pipeline_status (timestamp, component, status, message) VALUES
    (NOW(), 'rubix44_poller', 'idle', 'Initialized'),
    (NOW(), 'feature_extraction', 'idle', 'Initialized'),
    (NOW(), 'model_training', 'idle', 'Initialized'),
    (NOW(), 'web_interface', 'running', 'Initialized')
ON DUPLICATE KEY UPDATE timestamp=NOW();

-- Sample experiment for testing
INSERT INTO experiments (experiment_id, experiment_name, description, start_date, researcher_name, status)
VALUES ('EXP_2026_001', 'Lavender Detection Baseline', 'Initial experiments with lavender detection using rubix44 recordings', '2026-01-04', 'Bernd', 'active')
ON DUPLICATE KEY UPDATE experiment_name=experiment_name;

COMMIT;
