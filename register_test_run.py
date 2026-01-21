#!/usr/bin/env python3
"""
Register the current 24hr test run in MariaDB so it appears on the web interface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from db_connection import DatabaseConnection
from datetime import datetime, timedelta

def register_test_run():
    """Register the current orchestrator test run in MariaDB"""

    db = DatabaseConnection(backend='mariadb')

    # Test run details
    experiment_id = "test_24hr_20260121"
    experiment_name = "24-Hour Orchestrator Test"
    description = "Testing continuous learning orchestrator for 24 hours with relaxed duration requirements (10s min)"
    start_time = datetime(2026, 1, 21, 7, 48, 42)  # Actual start time
    end_time = start_time + timedelta(hours=24)
    orchestrator_pid = 69418  # Current PID

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # Check if already exists
        cursor.execute("""
            SELECT experiment_id FROM continuous_experiments
            WHERE experiment_id = %s
        """, (experiment_id,))

        if cursor.fetchone():
            print(f"Experiment {experiment_id} already exists, updating...")
            cursor.execute("""
                UPDATE continuous_experiments
                SET status = 'running',
                    orchestrator_pid = %s,
                    start_time = %s,
                    end_time = %s,
                    total_samples_collected = 60,
                    current_cycle = 1,
                    updated_at = NOW()
                WHERE experiment_id = %s
            """, (orchestrator_pid, start_time, end_time, experiment_id))
        else:
            print(f"Creating new experiment {experiment_id}...")
            cursor.execute("""
                INSERT INTO continuous_experiments (
                    experiment_id,
                    experiment_name,
                    description,
                    start_time,
                    end_time,
                    target_duration_weeks,
                    recording_interval_minutes,
                    playback_file,
                    output_prefix,
                    recording_duration_seconds,
                    channel_1_substance,
                    channel_2_substance,
                    beaker_1_role,
                    beaker_1_content,
                    beaker_2_role,
                    beaker_2_content,
                    faraday_cage_used,
                    researcher_name,
                    auto_qc_enabled,
                    training_sliding_window_weeks,
                    training_batch_size,
                    training_epochs_per_cycle,
                    training_learning_rate,
                    status,
                    orchestrator_pid,
                    current_cycle,
                    total_cycles_expected,
                    next_cycle_scheduled,
                    total_samples_collected,
                    created_at,
                    updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    0.14, 5, 'test_recording', 'test_24hr', 180,
                    'empty', 'empty', 'negative', 'empty', 'negative', 'empty',
                    0, 'Claude', 0,
                    4, 32, 5, 0.0001,
                    'running', %s, 1, 288, %s,
                    60,
                    NOW(), NOW()
                )
            """, (
                experiment_id, experiment_name, description, start_time, end_time,
                orchestrator_pid, start_time + timedelta(minutes=5)
            ))

        conn.commit()
        print(f"✓ Successfully registered {experiment_id} in MariaDB")
        print(f"  Status: running")
        print(f"  PID: {orchestrator_pid}")
        print(f"  Start: {start_time}")
        print(f"  End: {end_time}")
        print(f"  Samples: 60")
        print(f"\nYou should now see this experiment on the web interface!")

if __name__ == '__main__':
    register_test_run()
