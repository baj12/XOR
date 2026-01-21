#!/usr/bin/env python
"""
Create and start a new continuous recording based on test empty5 settings
"""
import sys
import json
import uuid
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from db_connection import DatabaseConnection


def create_experiment_from_empty5():
    """
    Create new experiment with test empty5 settings
    """
    # Settings from test empty5
    settings = {
        'experiment_name': f'Continuous Recording {Path(__file__).stem}',
        'target_duration_weeks': 0.03,  # ~30 minutes
        'recording_interval_minutes': 6,
        'playback_file': 'exp6 - 3 noise.wav',
        'output_prefix': 'continuous',
        'recording_duration_seconds': 180,  # 3 minutes
        'channel_1_substance': 'air',
        'channel_2_substance': 'lavander',
        'beaker_1_role': 'none',
        'beaker_1_content': '',
        'beaker_2_role': 'none',
        'beaker_2_content': '',
        'faraday_cage_used': True,
        'researcher_name': 'Autonomous',
        'auto_qc_enabled': True,
        'auto_qc_min_separation_score': 0.7,
        'auto_qc_min_samples_per_channel': 900,
        'auto_qc_auto_approve_threshold': 0.8,
        'auto_qc_auto_reject_threshold': 0.6,
        'training_sliding_window_weeks': 2,
        'training_batch_size': 32,
        'training_epochs_per_cycle': 5,
        'training_learning_rate': 0.0001
    }

    # Generate experiment ID
    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

    # Calculate total cycles
    total_minutes = settings['target_duration_weeks'] * 7 * 24 * 60
    interval_minutes = settings['recording_interval_minutes']
    total_cycles = int(total_minutes / interval_minutes)

    print(f"Creating new experiment: {experiment_id}")
    print(f"  Name: {settings['experiment_name']}")
    print(f"  Duration: {settings['target_duration_weeks']} weeks (~{total_minutes:.0f} minutes)")
    print(f"  Recording interval: {interval_minutes} minutes")
    print(f"  Recording duration: {settings['recording_duration_seconds']} seconds")
    print(f"  Total cycles: {total_cycles}")
    print(f"  Channel 1: {settings['channel_1_substance']}")
    print(f"  Channel 2: {settings['channel_2_substance']}")
    print(f"  Playback file: {settings['playback_file']}")

    db = DatabaseConnection(backend='mariadb')

    with db.get_connection() as conn:
        cursor = conn.cursor(buffered=True)

        # Insert experiment
        cursor.execute("""
            INSERT INTO continuous_experiments
            (experiment_id, experiment_name, target_duration_weeks, recording_interval_minutes,
             playback_file, recording_duration_seconds, output_prefix,
             channel_1_substance, channel_2_substance,
             beaker_1_role, beaker_1_content,
             beaker_2_role, beaker_2_content,
             faraday_cage_used, researcher_name,
             auto_qc_enabled, auto_qc_min_separation_score,
             auto_qc_auto_approve_threshold, auto_qc_auto_reject_threshold,
             auto_qc_min_samples_per_channel,
             training_sliding_window_weeks, training_batch_size, training_epochs_per_cycle,
             training_learning_rate,
             status, current_cycle, total_cycles_expected, created_at, start_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (
            experiment_id,
            settings['experiment_name'],
            settings['target_duration_weeks'],
            settings['recording_interval_minutes'],
            settings['playback_file'],
            settings['recording_duration_seconds'],
            settings['output_prefix'],
            settings['channel_1_substance'],
            settings['channel_2_substance'],
            settings['beaker_1_role'],
            settings['beaker_1_content'],
            settings['beaker_2_role'],
            settings['beaker_2_content'],
            settings['faraday_cage_used'],
            settings['researcher_name'],
            settings['auto_qc_enabled'],
            settings['auto_qc_min_separation_score'],
            settings['auto_qc_auto_approve_threshold'],
            settings['auto_qc_auto_reject_threshold'],
            settings['auto_qc_min_samples_per_channel'],
            settings['training_sliding_window_weeks'],
            settings['training_batch_size'],
            settings['training_epochs_per_cycle'],
            settings['training_learning_rate'],
            'stopped',  # Initial status
            0,  # current_cycle
            total_cycles
        ))

        conn.commit()

    print(f"\n✓ Experiment created: {experiment_id}")
    return experiment_id


def start_experiment(experiment_id):
    """
    Start the experiment by launching the recording orchestrator
    """
    import subprocess
    import sys
    from pathlib import Path

    # Get orchestrator path
    orchestrator_path = Path(__file__).parent.parent / 'src' / 'continuous' / 'recording_orchestrator.py'

    # Create log directory
    log_dir = Path(__file__).parent.parent / 'logs' / 'continuous'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_id}.log"

    print(f"\nStarting experiment orchestrator...")
    print(f"  Log file: {log_file}")

    # Start orchestrator as background process
    with open(log_file, 'w') as log_handle:
        process = subprocess.Popen(
            [sys.executable, str(orchestrator_path), experiment_id, '--log-level', 'INFO'],
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True  # Detach from parent
        )

    # Update database with PID
    db = DatabaseConnection(backend='mariadb')
    with db.get_connection() as conn:
        cursor = conn.cursor(buffered=True)
        cursor.execute("""
            UPDATE continuous_experiments
            SET orchestrator_pid = %s,
                status = 'running',
                start_time = NOW(),
                updated_at = NOW()
            WHERE experiment_id = %s
        """, (process.pid, experiment_id))
        conn.commit()

    print(f"✓ Orchestrator started (PID: {process.pid})")
    print(f"\nMonitor progress:")
    print(f"  tail -f {log_file}")
    print(f"\nOr via web interface:")
    print(f"  http://localhost:5001/continuous")
    print(f"\nStop experiment:")
    print(f"  kill {process.pid}")
    print(f"  # or via web interface")

    return process.pid


if __name__ == '__main__':
    try:
        # Create experiment
        experiment_id = create_experiment_from_empty5()

        # Start it
        pid = start_experiment(experiment_id)

        print(f"\n{'='*60}")
        print(f"Experiment ID: {experiment_id}")
        print(f"PID: {pid}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
