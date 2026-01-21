"""
Initialize MariaDB database for XOR project.

Creates the database and all required tables as defined in docs/MARIADB_SCHEMA.md
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConfig, create_database
import mysql.connector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tables(config: DatabaseConfig, db_name: str = 'xor_project'):
    """Create all tables for the XOR project"""

    logger.info(f"Creating tables in database: {db_name}")

    conn = mysql.connector.connect(
        host=config.mariadb_host,
        port=config.mariadb_port,
        user=config.mariadb_user,
        password=config.mariadb_password,
        database=db_name,
        ssl_disabled=True
    )

    cursor = conn.cursor()

    # 1. Features table (continuous learning)
    logger.info("Creating features table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 2. Training runs table
    logger.info("Creating training_runs table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
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
            INDEX idx_model_path (model_path(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 3. Validation metrics table
    logger.info("Creating validation_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_metrics (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 4. Drift metrics table
    logger.info("Creating drift_metrics table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_metrics (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 5. Experiments table
    logger.info("Creating experiments table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 6. GA runs table
    logger.info("Creating ga_runs table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ga_runs (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 7. GA individuals table
    logger.info("Creating ga_individuals table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ga_individuals (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 8. Model results table
    logger.info("Creating model_results table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 9. Resource usage table
    logger.info("Creating resource_usage table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resource_usage (
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    conn.commit()
    conn.close()

    logger.info("✅ All tables created successfully!")


def main():
    """Main function to initialize the database"""

    db_name = 'xor_project'

    print(f"\n{'='*60}")
    print(f"XOR Project - MariaDB Initialization")
    print(f"{'='*60}\n")

    # Step 1: Create database
    print(f"Step 1: Creating database '{db_name}'...")
    if create_database(backend='mariadb', db_name=db_name, drop_if_exists=False):
        print(f"✅ Database '{db_name}' created/exists\n")
    else:
        print(f"❌ Failed to create database '{db_name}'")
        return 1

    # Step 2: Create tables
    print("Step 2: Creating tables...")
    try:
        config = DatabaseConfig()
        create_tables(config, db_name)
        print("✅ All tables created successfully!\n")
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return 1

    # Step 3: Verify setup
    print("Step 3: Verifying setup...")
    try:
        from db_connection import DatabaseConnection
        db = DatabaseConnection(backend='mariadb')
        if db.test_connection():
            print("✅ Connection test successful!")
        else:
            print("❌ Connection test failed")
            return 1
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return 1

    print(f"\n{'='*60}")
    print("✅ MariaDB initialization complete!")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("1. Run migrations to import existing SQLite data")
    print("2. Update continuous learning modules to use MariaDB")
    print("3. Test end-to-end workflows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
