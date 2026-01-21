# MariaDB Schema Design for XOR Project

## Overview

This document describes the MariaDB database schema for storing experimental results, continuous learning features, and model training metadata.

## Database Configuration

- **Host**: 10.0.0.103
- **User**: bernd (or devuser)
- **Database**: xor_project
- **Connection**: SSL disabled for local network
- **Credentials**: Stored in `../.env` file

## Schema Design

### 1. Core Tables for Continuous Learning

#### `features`
Stores transformed audio features (not raw audio).

```sql
CREATE TABLE features (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    class_label TINYINT NOT NULL,
    channel TINYINT NOT NULL COMMENT '0=right/negative, 1=left/positive',
    source_file VARCHAR(255),
    segment_offset_sec DECIMAL(10,3),
    features MEDIUMBLOB NOT NULL COMMENT 'Compressed numpy array',
    extraction_version VARCHAR(20) DEFAULT 'v1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp),
    INDEX idx_class (class_label),
    INDEX idx_created (created_at),
    INDEX idx_source (source_file)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `training_runs`
Tracks all model training executions.

```sql
CREATE TABLE training_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_timestamp DATETIME NOT NULL,
    model_path VARCHAR(512) NOT NULL,
    data_start_date DATETIME,
    data_end_date DATETIME,
    n_training_samples INT,
    n_validation_samples INT,
    train_accuracy DECIMAL(6,5),
    val_accuracy DECIMAL(6,5),
    test_accuracy DECIMAL(6,5),
    roc_auc DECIMAL(6,5),
    update_mode VARCHAR(20) COMMENT 'incremental, full_retrain, hybrid',
    parent_model_id BIGINT,
    config_snapshot TEXT COMMENT 'JSON snapshot of config',
    duration_seconds DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_model_id) REFERENCES training_runs(id) ON DELETE SET NULL,
    INDEX idx_run_timestamp (run_timestamp),
    INDEX idx_model_path (model_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `validation_metrics`
Detailed validation metrics per training run.

```sql
CREATE TABLE validation_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    val_timestamp DATETIME NOT NULL,
    training_run_id BIGINT NOT NULL,
    accuracy DECIMAL(6,5),
    precision_score DECIMAL(6,5),
    recall_score DECIMAL(6,5),
    f1_score DECIMAL(6,5),
    roc_auc DECIMAL(6,5),
    true_negatives INT,
    false_positives INT,
    false_negatives INT,
    true_positives INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE,
    INDEX idx_val_timestamp (val_timestamp),
    INDEX idx_training_run (training_run_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `drift_metrics`
Tracks feature distribution drift over time.

```sql
CREATE TABLE drift_metrics (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    metric_timestamp DATETIME NOT NULL,
    analysis_start DATETIME,
    analysis_end DATETIME,
    kl_divergence DECIMAL(10,6),
    mean_shift_max DECIMAL(10,6),
    std_shift_max DECIMAL(10,6),
    class_0_percentage DECIMAL(5,2),
    class_1_percentage DECIMAL(5,2),
    feature_means TEXT COMMENT 'JSON array of feature means',
    feature_stds TEXT COMMENT 'JSON array of feature stds',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_timestamp (metric_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 2. Experimental Results Tables

#### `experiments`
Stores metadata for each XOR/audio experiment run.

```sql
CREATE TABLE experiments (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id VARCHAR(100) NOT NULL UNIQUE,
    config_file VARCHAR(255),
    description TEXT,
    experiment_type VARCHAR(50) COMMENT 'xor, audio, continuous',
    noise_dimensions INT,
    class_separation DECIMAL(5,3),
    dataset_size INT,
    started_at DATETIME,
    completed_at DATETIME,
    status VARCHAR(20) COMMENT 'pending, running, completed, failed',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_experiment_id (experiment_id),
    INDEX idx_status (status),
    INDEX idx_type (experiment_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `ga_runs`
Genetic algorithm execution data.

```sql
CREATE TABLE ga_runs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id BIGINT NOT NULL,
    population_size INT,
    crossover_prob DECIMAL(4,3),
    mutation_prob DECIMAL(4,3),
    num_generations INT,
    max_time_per_individual INT,
    epochs_per_individual INT,
    n_processes INT,
    started_at DATETIME,
    completed_at DATETIME,
    best_fitness DECIMAL(10,6),
    final_generation INT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
    INDEX idx_experiment (experiment_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `ga_individuals`
Stores individual solutions evaluated by GA.

```sql
CREATE TABLE ga_individuals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    ga_run_id BIGINT NOT NULL,
    generation INT NOT NULL,
    individual_index INT NOT NULL,
    hyperparameters TEXT COMMENT 'JSON of learning_rate, batch_size, etc',
    fitness_accuracy DECIMAL(6,5),
    fitness_loss DECIMAL(10,6),
    training_time_sec DECIMAL(10,3),
    model_architecture TEXT COMMENT 'JSON of layer config',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ga_run_id) REFERENCES ga_runs(id) ON DELETE CASCADE,
    INDEX idx_ga_run (ga_run_id),
    INDEX idx_generation (generation),
    INDEX idx_fitness (fitness_accuracy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

#### `model_results`
Final model performance metrics.

```sql
CREATE TABLE model_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id BIGINT NOT NULL,
    model_path VARCHAR(512),
    train_accuracy DECIMAL(6,5),
    test_accuracy DECIMAL(6,5),
    train_loss DECIMAL(10,6),
    test_loss DECIMAL(10,6),
    roc_auc DECIMAL(6,5),
    confusion_matrix TEXT COMMENT 'JSON 2x2 matrix',
    training_history TEXT COMMENT 'JSON of epoch-wise metrics',
    num_parameters BIGINT,
    hidden_layers TEXT COMMENT 'JSON array of layer sizes',
    skip_connections VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
    INDEX idx_experiment (experiment_id),
    INDEX idx_accuracy (test_accuracy)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 3. Resource Monitoring Tables

#### `resource_usage`
Tracks CPU, memory, and GPU usage during experiments.

```sql
CREATE TABLE resource_usage (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    experiment_id BIGINT,
    timestamp DATETIME NOT NULL,
    cpu_percent DECIMAL(5,2),
    memory_mb DECIMAL(10,2),
    gpu_utilization DECIMAL(5,2),
    gpu_memory_mb DECIMAL(10,2),
    process_count INT,
    thread_count INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
    INDEX idx_experiment (experiment_id),
    INDEX idx_timestamp (timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

## Advantages Over SQLite

1. **Concurrent Access**: Multiple processes can write simultaneously
2. **Scalability**: Better performance with large datasets (millions of features)
3. **Network Access**: Can query from multiple machines
4. **Better Indexing**: More sophisticated index types
5. **Transactions**: ACID compliance with InnoDB
6. **Backup**: Built-in tools for hot backups
7. **Monitoring**: Better query profiling and optimization tools

## Migration Strategy

1. Keep SQLite for local/dev work
2. Use MariaDB for production continuous learning
3. Add `--db-backend` flag to choose SQLite or MariaDB
4. Provide migration script to transfer existing SQLite data

## Connection Pooling

Use connection pooling for performance:
- Pool size: 5-10 connections
- Max overflow: 20
- Pool recycle: 3600 seconds (1 hour)

## Backup Strategy

- Daily automated backups via `mysqldump`
- Retention: 7 days rolling
- Weekly full backups retained for 4 weeks
- Store in separate backup server or cloud storage
