"""
Recording Management Web Interface

Flask application for managing recording sessions, metadata annotation,
quality control, and pipeline monitoring.

Usage:
    python web/app.py [--port 5001] [--debug]
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from datetime import datetime
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Import database connection
from db_connection import DatabaseConnection

# Import weather service and rubix44 directly to avoid loading TensorFlow
# (continuous.__init__.py imports all modules including tensorflow-dependent ones)
continuous_path = Path(__file__).parent.parent / 'src' / 'continuous'
sys.path.insert(0, str(continuous_path))
from weather_service import WeatherService
from rubix44_data_provider import Rubix44Client
sys.path.remove(str(continuous_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize services
db = DatabaseConnection(backend='mariadb')
weather_service = WeatherService()


# Cleanup function for stale experiments
def cleanup_stale_experiments_on_startup():
    """
    Cleanup experiments marked as 'running' but with no actual process.

    This handles cases where the system restarted/hibernated while experiments
    were running, leaving stale 'running' status in the database.
    """
    try:
        logger.info("Running stale experiment cleanup...")

        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Find all experiments marked as running
            cursor.execute("""
                SELECT experiment_id, experiment_name, orchestrator_pid, status
                FROM continuous_experiments
                WHERE status = 'running'
            """)
            running_experiments = cursor.fetchall()

            if not running_experiments:
                logger.info("No running experiments found - nothing to clean up")
                return

            cleaned = 0
            for exp in running_experiments:
                exp_id = exp['experiment_id']
                pid = exp['orchestrator_pid']

                # Check if process is actually running (not zombie)
                is_stale = False
                if pid is None:
                    is_stale = True
                    reason = "No PID stored"
                else:
                    try:
                        os.kill(pid, 0)  # Signal 0 just checks existence

                        # Process exists, but check if it's a zombie
                        import subprocess
                        result = subprocess.run(
                            ['ps', '-p', str(pid), '-o', 'stat='],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )

                        if result.returncode == 0:
                            stat = result.stdout.strip()
                            # If status starts with 'Z', it's a zombie (defunct)
                            if stat and stat[0] == 'Z':
                                is_stale = True
                                reason = f"Process {pid} is zombie/defunct"
                            else:
                                # Process exists and is not zombie - not stale
                                logger.info(f"✓ {exp_id}: Process {pid} is running")
                                continue
                        else:
                            is_stale = True
                            reason = f"Process {pid} not found"

                    except (OSError, ProcessLookupError):
                        is_stale = True
                        reason = f"Process {pid} not running"
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Timeout checking PID {pid}, assuming stale")
                        is_stale = True
                        reason = f"Process {pid} check timeout"
                    except Exception as e:
                        logger.warning(f"Error checking PID {pid}: {e}")
                        is_stale = True
                        reason = f"Process {pid} check failed: {e}"

                if is_stale:
                    logger.warning(f"⚠ Stale experiment detected: {exp_id} ({reason})")

                    # Update to stopped
                    cursor.execute("""
                        UPDATE continuous_experiments
                        SET status = 'stopped',
                            orchestrator_pid = NULL,
                            updated_at = NOW()
                        WHERE experiment_id = %s
                    """, (exp_id,))

                    # Log alert
                    import json
                    cursor.execute("""
                        INSERT INTO experiment_alerts
                        (experiment_id, alert_type, severity, message, details)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        exp_id,
                        'other',
                        'warning',
                        f'Stale process cleanup: Experiment marked as stopped due to missing process. {reason}',
                        json.dumps({
                            'experiment_id': exp_id,
                            'pid': pid,
                            'reason': reason,
                            'cleanup_time': datetime.now().isoformat()
                        })
                    ))

                    conn.commit()
                    cleaned += 1
                    logger.info(f"  → Cleaned up {exp_id}: status='stopped', PID cleared")

            if cleaned > 0:
                logger.info(f"✓ Cleanup complete: {cleaned} stale experiment(s) cleaned")
            else:
                logger.info("✓ All running experiments have valid processes")

    except Exception as e:
        logger.error(f"Error during stale experiment cleanup: {e}", exc_info=True)


# Setup periodic cleanup scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=cleanup_stale_experiments_on_startup,
    trigger='interval',
    minutes=5,
    id='stale_experiment_cleanup',
    name='Periodic stale experiment cleanup',
    replace_existing=True
)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

logger.info("Periodic stale experiment cleanup scheduled (every 5 minutes)")


@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/annotate')
def annotate():
    """Recording annotation interface"""
    return render_template('annotate.html')


@app.route('/recordings')
def recordings():
    """Recordings list page"""
    return render_template('recordings.html')


@app.route('/qc')
def qc():
    """Quality control visualization page"""
    return render_template('qc.html')


@app.route('/api/recordings')
def api_recordings_list():
    """
    API endpoint: Get list of all recordings

    Query parameters:
        - status: filter by metadata_complete/quality_approved
        - experiment_id: filter by experiment
        - limit: max results (default: 100)
        - offset: pagination offset (default: 0)
    """
    # Get query parameters
    status_filter = request.args.get('status', None)
    experiment_id = request.args.get('experiment_id', None)
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # Build query
            query = "SELECT * FROM recording_dashboard WHERE 1=1"
            params = []

            if status_filter == 'pending_metadata':
                query += " AND metadata_complete = FALSE"
            elif status_filter == 'pending_qc':
                query += " AND metadata_complete = TRUE AND quality_approved IS NULL"
            elif status_filter == 'approved':
                query += " AND quality_approved = TRUE"
            elif status_filter == 'rejected':
                query += " AND quality_approved = FALSE"

            if experiment_id:
                query += " AND experiment_id = %s"
                params.append(experiment_id)

            query += " ORDER BY recording_date DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            recordings = cursor.fetchall()

            # Convert datetime objects to strings
            for rec in recordings:
                for key, value in rec.items():
                    if isinstance(value, datetime):
                        rec[key] = value.isoformat()

            # Get total count
            count_query = "SELECT COUNT(*) as total FROM recording_sessions"
            cursor.execute(count_query)
            total = cursor.fetchone()['total']

            return jsonify({
                'success': True,
                'recordings': recordings,
                'total': total,
                'limit': limit,
                'offset': offset
            })

    except Exception as e:
        logger.error(f"Failed to fetch recordings: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rubix44/recordings')
def api_rubix44_recordings():
    """
    API endpoint: Get available recordings from rubix44-recorder
    """
    try:
        # Get rubix44 URL from config or use default
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
        client = Rubix44Client(rubix_url)

        # Get recording history
        history = client.get_recording_history()

        return jsonify({
            'success': True,
            'recordings': history,
            'api_url': rubix_url
        })

    except Exception as e:
        logger.error(f"Failed to fetch rubix44 recordings: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/weather')
def api_weather():
    """
    API endpoint: Get current weather for Viroflay
    """
    try:
        weather_data = weather_service.get_weather_dict()
        summary = weather_service.get_weather_summary()

        # Convert datetime to string
        if 'weather_timestamp' in weather_data:
            weather_data['weather_timestamp'] = weather_data['weather_timestamp'].isoformat()

        return jsonify({
            'success': True,
            'weather': weather_data,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Failed to fetch weather: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recordings/<session_id>', methods=['GET'])
def api_recording_get(session_id):
    """
    API endpoint: Get single recording details
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            query = "SELECT * FROM recording_sessions WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            recording = cursor.fetchone()

            if not recording:
                return jsonify({'success': False, 'error': 'Recording not found'}), 404

            # Convert datetime objects to strings
            for key, value in recording.items():
                if isinstance(value, datetime):
                    recording[key] = value.isoformat()

            return jsonify({
                'success': True,
                'recording': recording
            })

    except Exception as e:
        logger.error(f"Failed to fetch recording {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/recordings/<session_id>', methods=['POST', 'PUT'])
def api_recording_update(session_id):
    """
    API endpoint: Create or update recording metadata
    """
    try:
        data = request.json

        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Check if recording exists
            cursor.execute("SELECT id FROM recording_sessions WHERE session_id = %s", (session_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing
                update_fields = []
                params = []

                # Map form fields to database columns
                field_mapping = {
                    'channel_1_source': 'channel_1_source',
                    'channel_2_source': 'channel_2_source',
                    'channel_1_expected_class': 'channel_1_expected_class',
                    'channel_2_expected_class': 'channel_2_expected_class',
                    'beaker_1_role': 'beaker_1_role',
                    'beaker_2_role': 'beaker_2_role',
                    'beaker_3_role': 'beaker_3_role',
                    'beaker_1_content': 'beaker_1_content',
                    'beaker_2_content': 'beaker_2_content',
                    'beaker_3_content': 'beaker_3_content',
                    'faraday_cage_used': 'faraday_cage_used',
                    'experiment_description': 'experiment_description',
                    'experiment_id': 'experiment_id',
                    'researcher_name': 'researcher_name',
                    'comments': 'comments',
                    'metadata_complete': 'metadata_complete',
                    'quality_approved': 'quality_approved'
                }

                for form_field, db_field in field_mapping.items():
                    if form_field in data:
                        update_fields.append(f"{db_field} = %s")
                        params.append(data[form_field])

                if update_fields:
                    params.append(session_id)
                    query = f"UPDATE recording_sessions SET {', '.join(update_fields)} WHERE session_id = %s"
                    cursor.execute(query, params)
                    conn.commit()

                return jsonify({'success': True, 'message': 'Recording updated'})

            else:
                # Create new recording
                fields = ['session_id']
                values = [session_id]
                placeholders = ['%s']

                # Add provided fields
                for key, value in data.items():
                    if key in ['channel_1_source', 'channel_2_source', 'channel_1_expected_class',
                              'channel_2_expected_class', 'beaker_1_role', 'beaker_2_role', 'beaker_3_role',
                              'beaker_1_content', 'beaker_2_content', 'beaker_3_content',
                              'faraday_cage_used', 'experiment_description', 'experiment_id',
                              'researcher_name', 'comments', 'recording_date', 'duration_seconds',
                              'sample_rate', 'stereo_filename']:
                        fields.append(key)
                        values.append(value)
                        placeholders.append('%s')

                query = f"INSERT INTO recording_sessions ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
                cursor.execute(query, values)
                conn.commit()

                return jsonify({'success': True, 'message': 'Recording created'})

    except Exception as e:
        logger.error(f"Failed to update recording {session_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pipeline/status')
def api_pipeline_status():
    """
    API endpoint: Get current pipeline status
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            # Get latest status for each component
            query = """
                SELECT component, status, message, timestamp, metrics
                FROM pipeline_status
                WHERE (component, timestamp) IN (
                    SELECT component, MAX(timestamp)
                    FROM pipeline_status
                    GROUP BY component
                )
                ORDER BY component
            """
            cursor.execute(query)
            statuses = cursor.fetchall()

            # Convert datetime to string
            for status in statuses:
                if status['timestamp']:
                    status['timestamp'] = status['timestamp'].isoformat()

            return jsonify({
                'success': True,
                'statuses': statuses
            })

    except Exception as e:
        logger.error(f"Failed to fetch pipeline status: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/experiments')
def api_experiments_list():
    """
    API endpoint: Get list of experiments
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            query = "SELECT * FROM experiments ORDER BY id DESC"
            cursor.execute(query)
            experiments = cursor.fetchall()

            # Convert datetime objects to strings
            for exp in experiments:
                for key, value in exp.items():
                    if isinstance(value, datetime):
                        exp[key] = value.isoformat()
                    elif hasattr(value, 'date'):  # DATE type
                        exp[key] = value.isoformat()

            return jsonify({
                'success': True,
                'experiments': experiments
            })

    except Exception as e:
        logger.error(f"Failed to fetch experiments: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rubix44/playback-files')
def api_rubix44_playback_files():
    """
    API endpoint: Get available playback files from rubix44-recorder
    """
    try:
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')

        # Make direct request
        import requests
        response = requests.get(f"{rubix_url}/api/v1/playback-files", timeout=5)
        response.raise_for_status()
        files = response.json()

        return jsonify({
            'success': True,
            'files': files
        })

    except Exception as e:
        logger.error(f"Failed to fetch playback files: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rubix44/status')
def api_rubix44_status():
    """
    API endpoint: Get rubix44 recording status with time estimation
    """
    try:
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
        client = Rubix44Client(rubix_url)

        status = client.get_recording_status()

        # Add time estimation if recording is active
        if status.get('status') == 'recording':
            elapsed = status.get('elapsed_seconds', 0)
            duration = status.get('duration', 0)  # rubix44 uses 'duration' not 'duration_seconds'

            if duration > 0:
                # Calculate remaining time
                remaining = max(0, duration - elapsed)
                progress_percent = (elapsed / duration) * 100

                # Add estimation fields
                status['time_remaining_seconds'] = remaining
                status['progress_percent'] = progress_percent

                # Format time strings for display
                status['elapsed_formatted'] = format_duration(elapsed)
                status['remaining_formatted'] = format_duration(remaining)
                status['duration_formatted'] = format_duration(duration)

        return jsonify(status)

    except Exception as e:
        logger.error(f"Failed to get rubix44 status: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def format_duration(seconds):
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    elif seconds < 86400:  # Less than 24 hours
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:  # 24 hours or more
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        if hours > 0:
            return f"{days}d {hours}h"
        else:
            return f"{days}d"


@app.route('/api/rubix44/start', methods=['POST'])
def api_rubix44_start():
    """
    API endpoint: Start a new recording on rubix44
    """
    try:
        data = request.json
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')

        # Make direct request to rubix44 API
        import requests
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={
                'playback_file': data.get('playback_file'),
                'duration': data.get('duration', 60),
                'output_prefix': data.get('output_prefix', 'recording')
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        return jsonify({
            'success': True,
            'session_id': result.get('session_id'),
            'message': 'Recording started'
        })

    except Exception as e:
        logger.error(f"Failed to start recording: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/rubix44/stop', methods=['POST'])
def api_rubix44_stop():
    """
    API endpoint: Stop current recording on rubix44
    """
    try:
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')

        import requests
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/stop",
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        return jsonify({
            'success': True,
            'message': 'Recording stopped',
            'files': result.get('files', [])
        })

    except Exception as e:
        logger.error(f"Failed to stop recording: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/qc/extract-features/<session_id>', methods=['POST'])
def api_qc_extract_features(session_id):
    """
    API endpoint: Extract features from a recording for QC visualization

    This endpoint:
    1. Downloads the recording from rubix44 (if needed)
    2. Extracts features using StereoChannelProcessor
    3. Performs dimensionality reduction (UMAP, t-SNE, PCA)
    4. Returns reduced features for visualization
    """
    try:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        try:
            from umap import UMAP
            has_umap = True
        except ImportError:
            has_umap = False
            logger.warning("UMAP not available, skipping UMAP visualization")

        # Get recording metadata
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)

            query = "SELECT * FROM recording_sessions WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            recording = cursor.fetchone()

            if not recording:
                return jsonify({'success': False, 'error': 'Recording not found'}), 404

            if not recording.get('metadata_complete'):
                return jsonify({'success': False, 'error': 'Metadata not complete'}), 400

        # Download WAV file from rubix44 if needed
        rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
        rubix_client = Rubix44Client(rubix_url)

        # Get session files
        history = rubix_client.get_recording_history()
        session_data = next((s for s in history if s['id'] == session_id), None)

        if not session_data:
            return jsonify({'success': False, 'error': 'Session not found on rubix44'}), 404

        # Find stereo file
        stereo_file = next((f for f in session_data.get('files', [])
                           if '_stereo.wav' in f['name']), None)

        if not stereo_file:
            return jsonify({'success': False, 'error': 'No stereo file found'}), 404

        # Download to temporary location
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        local_path = temp_dir / stereo_file['name']

        logger.info(f"Downloading {stereo_file['name']} for QC...")
        if not rubix_client.download_recording(stereo_file['name'], local_path):
            return jsonify({'success': False, 'error': 'Failed to download file'}), 500

        # Extract features using StereoChannelProcessor
        from continuous.stereo_channel_processor import StereoChannelProcessor
        from utils import Config

        # Create minimal config for processor
        config = Config.from_yaml(Path(__file__).parent.parent / 'config' / 'audio_config.yaml')
        processor = StereoChannelProcessor(config)

        logger.info(f"Extracting features from {stereo_file['name']}...")
        # Limit to 1000 samples per channel for QC (faster processing)
        X_left, y_left, X_right, y_right, _, _ = processor.process_stereo_file(
            local_path,
            max_samples_per_channel=1000,
            positive_label=int(recording['channel_1_expected_class']),
            negative_label=int(recording['channel_2_expected_class'])
        )

        # Combine features and labels
        X = np.vstack([X_left, X_right])
        y = np.concatenate([y_left, y_right])

        logger.info(f"Performing dimensionality reduction on {len(X)} samples...")

        # Perform dimensionality reduction
        results = {
            'session_id': session_id,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_counts': {
                str(recording['channel_1_expected_class']): int(np.sum(y == recording['channel_1_expected_class'])),
                str(recording['channel_2_expected_class']): int(np.sum(y == recording['channel_2_expected_class']))
            }
        }

        # PCA (dimensions 1-2 and 3-4)
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X)
        results['pca'] = {
            'coords_1_2': X_pca[:, :2].tolist(),
            'coords_3_4': X_pca[:, 2:4].tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'labels': y.tolist()
        }

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) // 4))
        X_tsne = tsne.fit_transform(X)
        results['tsne'] = {
            'coords': X_tsne.tolist(),
            'labels': y.tolist()
        }

        # UMAP (if available)
        if has_umap:
            umap_model = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(X) // 4))
            X_umap = umap_model.fit_transform(X)
            results['umap'] = {
                'coords': X_umap.tolist(),
                'labels': y.tolist()
            }

        # Calculate class separation metrics
        from sklearn.metrics import silhouette_score
        from scipy.spatial.distance import cdist

        # Silhouette score (higher is better, range -1 to 1)
        if len(np.unique(y)) > 1:
            silhouette = float(silhouette_score(X_pca[:, :2], y))
            results['metrics'] = {
                'silhouette_score': silhouette,
                'interpretation': 'Good' if silhouette > 0.5 else 'Fair' if silhouette > 0.25 else 'Poor'
            }

            # Inter-class distance vs intra-class distance
            class_0_idx = y == recording['channel_2_expected_class']
            class_1_idx = y == recording['channel_1_expected_class']

            if np.any(class_0_idx) and np.any(class_1_idx):
                # Mean distance between classes
                inter_dist = np.mean(cdist(X_pca[class_0_idx, :2], X_pca[class_1_idx, :2]))

                # Mean distance within classes
                intra_dist_0 = np.mean(cdist(X_pca[class_0_idx, :2], X_pca[class_0_idx, :2])) if np.sum(class_0_idx) > 1 else 0
                intra_dist_1 = np.mean(cdist(X_pca[class_1_idx, :2], X_pca[class_1_idx, :2])) if np.sum(class_1_idx) > 1 else 0
                intra_dist = (intra_dist_0 + intra_dist_1) / 2

                separation_ratio = inter_dist / (intra_dist + 1e-10)
                results['metrics']['separation_ratio'] = float(separation_ratio)
                results['metrics']['separation_quality'] = 'Excellent' if separation_ratio > 2.0 else 'Good' if separation_ratio > 1.5 else 'Fair' if separation_ratio > 1.0 else 'Poor'

        # Cleanup temporary file
        local_path.unlink()
        temp_dir.rmdir()

        logger.info(f"QC feature extraction complete for {session_id}")

        return jsonify({
            'success': True,
            'data': results
        })

    except Exception as e:
        logger.error(f"Failed to extract features for QC: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/qc/approve/<session_id>', methods=['POST'])
def api_qc_approve(session_id):
    """
    API endpoint: Approve recording quality
    """
    try:
        data = request.json or {}
        qc_notes = data.get('notes', '')

        with db.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                UPDATE recording_sessions
                SET quality_approved = TRUE,
                    qc_notes = %s
                WHERE session_id = %s
            """
            cursor.execute(query, (qc_notes, session_id))
            conn.commit()

            logger.info(f"Approved QC for {session_id}")

            return jsonify({
                'success': True,
                'message': 'Recording approved for processing'
            })

    except Exception as e:
        logger.error(f"Failed to approve QC: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/qc/reject/<session_id>', methods=['POST'])
def api_qc_reject(session_id):
    """
    API endpoint: Reject recording quality
    """
    try:
        data = request.json or {}
        qc_notes = data.get('notes', '')

        if not qc_notes:
            return jsonify({'success': False, 'error': 'Rejection reason required'}), 400

        with db.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                UPDATE recording_sessions
                SET quality_approved = FALSE,
                    qc_notes = %s
                WHERE session_id = %s
            """
            cursor.execute(query, (qc_notes, session_id))
            conn.commit()

            logger.info(f"Rejected QC for {session_id}: {qc_notes}")

            return jsonify({
                'success': True,
                'message': 'Recording rejected'
            })

    except Exception as e:
        logger.error(f"Failed to reject QC: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# ========================================================================
# PIPELINE MONITOR ROUTES
# ========================================================================

@app.route('/pipeline')
def pipeline_monitor():
    """Pipeline monitoring dashboard page"""
    return render_template('pipeline_monitor.html')


# ========================================================================
# TRAINING ROUTES
# ========================================================================

@app.route('/training')
def training_dashboard():
    """Training dashboard page"""
    return render_template('training_dashboard.html')


# ========================================================================
# CONTINUOUS EXPERIMENTS ROUTES
# ========================================================================

@app.route('/continuous')
def continuous_home():
    """Continuous experiments dashboard page"""
    return render_template('continuous_dashboard.html')


@app.route('/continuous/create')
def continuous_create():
    """Create new continuous experiment page"""
    return render_template('continuous_experiment.html')


@app.route('/api/continuous/experiments', methods=['GET'])
def api_continuous_experiments_list():
    """
    API endpoint: Get list of all continuous experiments

    Query parameters:
        - status: filter by status (running/paused/completed/failed/stopped)
        - limit: max results (default: 50)
        - offset: pagination offset (default: 0)
    """
    status_filter = request.args.get('status', None)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Build query
            query = "SELECT * FROM experiment_summary WHERE 1=1"
            params = []

            if status_filter:
                query += " AND status = %s"
                params.append(status_filter)

            query += " ORDER BY start_time DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            experiments = cursor.fetchall()

            # Convert datetime objects to strings
            for exp in experiments:
                for key, value in exp.items():
                    if isinstance(value, datetime):
                        exp[key] = value.isoformat()

            # Get total count
            count_query = "SELECT COUNT(*) as total FROM continuous_experiments"
            if status_filter:
                count_query += " WHERE status = %s"
                cursor.execute(count_query, [status_filter])
            else:
                cursor.execute(count_query)
            total = cursor.fetchone()['total']

            return jsonify({
                'success': True,
                'experiments': experiments,
                'total': total,
                'limit': limit,
                'offset': offset
            })

    except Exception as e:
        logger.error(f"Failed to fetch continuous experiments: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments', methods=['POST'])
def api_continuous_experiments_create():
    """
    API endpoint: Create new continuous experiment

    Required fields:
        - experiment_name: string
        - target_duration_weeks: float (supports fractional weeks, e.g., 0.01 = ~10 minutes)
        - recording_interval_minutes: int (default: 60)
        - playback_file: string
        - channel_1_substance, channel_2_substance: string (from vocabulary)
    """
    try:
        data = request.json

        # Import substance vocabulary (direct import to avoid TensorFlow)
        import sys
        from pathlib import Path
        import importlib.util

        vocab_path = Path(__file__).parent.parent / 'src' / 'continuous' / 'substance_vocabulary.py'
        spec = importlib.util.spec_from_file_location("substance_vocabulary", vocab_path)
        vocab_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vocab_module)

        validate_substance = vocab_module.validate_substance
        get_class_for_substance = vocab_module.get_class_for_substance

        # Validate required fields
        required = ['experiment_name', 'target_duration_weeks', 'playback_file',
                   'channel_1_substance', 'channel_2_substance']

        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Validate target_duration_weeks
        try:
            target_duration = float(data['target_duration_weeks'])
            if target_duration <= 0:
                return jsonify({'success': False, 'error': 'target_duration_weeks must be greater than 0'}), 400
            if target_duration > 1000:
                return jsonify({'success': False, 'error': 'target_duration_weeks must be less than 1000'}), 400
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'target_duration_weeks must be a valid number'}), 400

        # Validate substances
        for channel in ['channel_1_substance', 'channel_2_substance']:
            is_valid, error_msg = validate_substance(data[channel])
            if not is_valid:
                return jsonify({'success': False, 'error': error_msg}), 400

        # Generate experiment ID
        import uuid
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

        with db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)

            # Calculate total cycles
            total_minutes = data['target_duration_weeks'] * 7 * 24 * 60
            interval_minutes = data.get('recording_interval_minutes', 60)
            total_cycles = int(total_minutes / interval_minutes)

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
                 status, current_cycle, total_cycles_expected, start_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                experiment_id,
                data['experiment_name'],
                data['target_duration_weeks'],
                interval_minutes,
                data['playback_file'],
                data.get('recording_duration_seconds', 3600),
                data.get('output_prefix', 'continuous'),
                data['channel_1_substance'],
                data['channel_2_substance'],
                data.get('beaker_1_role', 'not_used'),
                data.get('beaker_1_content', ''),
                data.get('beaker_2_role', 'not_used'),
                data.get('beaker_2_content', ''),
                data.get('faraday_cage_used', True),  # Now defaults to True
                data.get('researcher_name', 'Autonomous'),
                data.get('auto_qc_enabled', True),
                data.get('auto_qc_min_separation_score', 0.7),
                data.get('auto_qc_auto_approve_threshold', 0.8),
                data.get('auto_qc_auto_reject_threshold', 0.6),
                data.get('auto_qc_min_samples_per_channel', 900),
                data.get('training_sliding_window_weeks', 2),
                data.get('training_batch_size', 32),
                data.get('training_epochs_per_cycle', 5),
                'stopped',  # Initial status
                0,  # current_cycle
                total_cycles
            ))
            conn.commit()

        logger.info(f"Created continuous experiment: {experiment_id}")

        return jsonify({
            'success': True,
            'experiment_id': experiment_id,
            'message': 'Experiment created successfully'
        })

    except Exception as e:
        logger.error(f"Failed to create continuous experiment: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/substances', methods=['GET'])
def api_substance_vocabulary():
    """
    API endpoint: Get substance vocabulary for UI selection

    Returns list of valid substances with their class labels and display names
    """
    try:
        import sys
        from pathlib import Path
        import importlib.util

        # Direct module import to avoid continuous/__init__.py
        vocab_path = Path(__file__).parent.parent / 'src' / 'continuous' / 'substance_vocabulary.py'
        spec = importlib.util.spec_from_file_location("substance_vocabulary", vocab_path)
        vocab_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vocab_module)

        get_substance_choices = vocab_module.get_substance_choices

        choices = get_substance_choices()

        # Format for JSON response
        substances = [
            {
                'name': name,
                'class': cls,
                'display': display
            }
            for name, cls, display in choices
        ]

        return jsonify({
            'success': True,
            'substances': substances
        })

    except Exception as e:
        logger.error(f"Failed to get substance vocabulary: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>', methods=['GET'])
def api_continuous_experiment_get(experiment_id):
    """
    API endpoint: Get single continuous experiment details
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Get experiment details
            cursor.execute("""
                SELECT * FROM continuous_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            experiment = cursor.fetchone()

            if not experiment:
                return jsonify({'success': False, 'error': 'Experiment not found'}), 404

            # Convert datetime objects to strings
            for key, value in experiment.items():
                if isinstance(value, datetime):
                    experiment[key] = value.isoformat()

            # Get recent cycles
            cursor.execute("""
                SELECT * FROM recording_cycles
                WHERE experiment_id = %s
                ORDER BY cycle_number DESC
                LIMIT 20
            """, (experiment_id,))

            cycles = cursor.fetchall()

            for cycle in cycles:
                for key, value in cycle.items():
                    if isinstance(value, datetime):
                        cycle[key] = value.isoformat()

            # Get alerts
            cursor.execute("""
                SELECT * FROM experiment_alerts
                WHERE experiment_id = %s
                AND acknowledged = FALSE
                ORDER BY created_at DESC
                LIMIT 10
            """, (experiment_id,))

            alerts = cursor.fetchall()

            for alert in alerts:
                for key, value in alert.items():
                    if isinstance(value, datetime):
                        alert[key] = value.isoformat()

            return jsonify({
                'success': True,
                'experiment': experiment,
                'recent_cycles': cycles,
                'unacknowledged_alerts': alerts
            })

    except Exception as e:
        logger.error(f"Failed to fetch experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/start', methods=['POST'])
def api_continuous_experiment_start(experiment_id):
    """
    API endpoint: Start a continuous experiment

    This spawns the orchestrator in a background process.
    """
    try:
        import subprocess
        import sys

        # Check if experiment exists
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)
            cursor.execute("""
                SELECT status FROM continuous_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            experiment = cursor.fetchone()

            if not experiment:
                return jsonify({'success': False, 'error': 'Experiment not found'}), 404

            if experiment['status'] == 'running':
                return jsonify({'success': False, 'error': 'Experiment already running'}), 400

        # Launch orchestrator in background
        orchestrator_path = Path(__file__).parent.parent / 'src' / 'continuous' / 'recording_orchestrator.py'
        log_dir = Path(__file__).parent.parent / 'logs' / 'continuous'
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{experiment_id}.log"

        # Start orchestrator as daemon process
        with open(log_file, 'a') as log_handle:
            process = subprocess.Popen(
                [sys.executable, str(orchestrator_path), experiment_id, '--log-level', 'INFO'],
                stdout=log_handle,
                stderr=log_handle,
                start_new_session=True  # Detach from parent
            )

        logger.info(f"Started continuous experiment {experiment_id} (PID: {process.pid})")

        # Store PID in database for process management
        with db.get_connection() as conn:
            cursor = conn.cursor(buffered=True)
            cursor.execute("""
                UPDATE continuous_experiments
                SET orchestrator_pid = %s,
                    status = 'running',
                    updated_at = NOW()
                WHERE experiment_id = %s
            """, (process.pid, experiment_id))
            conn.commit()

        return jsonify({
            'success': True,
            'message': 'Experiment started',
            'pid': process.pid,
            'log_file': str(log_file)
        })

    except Exception as e:
        logger.error(f"Failed to start experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/pause', methods=['POST'])
def api_continuous_experiment_pause(experiment_id):
    """
    API endpoint: Pause a running continuous experiment

    Sets the should_stop flag by updating database status.
    The orchestrator will detect this and pause gracefully.
    """
    try:
        import os
        import signal
        import time

        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Get current PID
            cursor.execute("""
                SELECT orchestrator_pid FROM continuous_experiments
                WHERE experiment_id = %s
                AND status = 'running'
            """, (experiment_id,))

            result = cursor.fetchone()
            if not result:
                return jsonify({'success': False, 'error': 'Experiment not running'}), 400

            pid = result['orchestrator_pid']

            # Set pause status (orchestrator will see this and stop)
            cursor.execute("""
                UPDATE continuous_experiments
                SET status = 'paused',
                    updated_at = NOW()
                WHERE experiment_id = %s
            """, (experiment_id,))

            conn.commit()

        logger.info(f"Paused experiment {experiment_id}, orchestrator PID: {pid}")

        # Check if process is actually running
        process_running = False
        if pid:
            try:
                os.kill(pid, 0)  # Signal 0 just checks if process exists
                process_running = True
            except (OSError, ProcessLookupError):
                process_running = False

        return jsonify({
            'success': True,
            'message': 'Experiment pause requested (will stop after current cycle)',
            'pid': pid,
            'process_still_running': process_running
        })

    except Exception as e:
        logger.error(f"Failed to pause experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/kill', methods=['POST'])
def api_continuous_experiment_kill(experiment_id):
    """
    API endpoint: Force kill a continuous experiment

    Sends SIGTERM to the orchestrator process and updates database.
    Use this when pause doesn't work or for immediate stop.
    """
    try:
        import os
        import signal

        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Get current PID and status
            cursor.execute("""
                SELECT orchestrator_pid, status FROM continuous_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            result = cursor.fetchone()
            if not result:
                return jsonify({'success': False, 'error': 'Experiment not found'}), 404

            pid = result['orchestrator_pid']
            current_status = result['status']

            if not pid:
                return jsonify({'success': False, 'error': 'No PID found - experiment may not be running'}), 400

            # Try to kill the process
            killed = False
            error_msg = None

            try:
                # First check if process exists
                os.kill(pid, 0)

                # Process exists, send SIGTERM (graceful shutdown)
                os.kill(pid, signal.SIGTERM)
                killed = True
                logger.info(f"Sent SIGTERM to orchestrator PID {pid} for experiment {experiment_id}")

            except ProcessLookupError:
                # Process doesn't exist
                error_msg = f"Process {pid} not found - may have already stopped"
                logger.warning(f"{error_msg} for experiment {experiment_id}")
                killed = False

            except PermissionError:
                error_msg = f"Permission denied to kill process {pid}"
                logger.error(f"{error_msg} for experiment {experiment_id}")
                return jsonify({'success': False, 'error': error_msg}), 403

            except Exception as e:
                error_msg = f"Failed to kill process {pid}: {str(e)}"
                logger.error(f"{error_msg} for experiment {experiment_id}")
                return jsonify({'success': False, 'error': error_msg}), 500

            # Update database status regardless
            cursor.execute("""
                UPDATE continuous_experiments
                SET status = 'stopped',
                    orchestrator_pid = NULL,
                    updated_at = NOW()
                WHERE experiment_id = %s
            """, (experiment_id,))

            conn.commit()

        return jsonify({
            'success': True,
            'message': 'Experiment killed' if killed else 'Experiment status updated (process already dead)',
            'pid': pid,
            'killed': killed,
            'previous_status': current_status,
            'warning': error_msg
        })

    except Exception as e:
        logger.error(f"Failed to kill experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/duplicate', methods=['POST'])
def api_continuous_experiment_duplicate(experiment_id):
    """
    API endpoint: Duplicate (re-run) a continuous experiment with automatic name update

    Creates a new experiment with the same configuration as the source experiment,
    automatically incrementing the name (e.g., "Exp A" -> "Exp A (2)").

    Optional JSON body:
        - experiment_name: Override the auto-generated name
        - auto_start: Boolean, whether to start immediately (default: false)
    """
    try:
        import re
        import uuid

        # Get optional parameters from request
        data = request.json or {}
        custom_name = data.get('experiment_name', None)
        auto_start = data.get('auto_start', False)

        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Get source experiment
            cursor.execute("""
                SELECT * FROM continuous_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            source_exp = cursor.fetchone()

            if not source_exp:
                return jsonify({'success': False, 'error': 'Source experiment not found'}), 404

            # Generate new experiment name
            if custom_name:
                new_name = custom_name
            else:
                base_name = source_exp['experiment_name']

                # Check if name already has a run number in parentheses
                match = re.search(r'^(.*?)\s*\((\d+)\)$', base_name)
                if match:
                    # Extract base name and increment number
                    name_base = match.group(1)
                    run_number = int(match.group(2)) + 1
                else:
                    # No run number yet, add (2)
                    name_base = base_name
                    run_number = 2

                new_name = f"{name_base} ({run_number})"

                # Ensure name is unique by checking database
                while True:
                    cursor.execute("""
                        SELECT COUNT(*) as cnt FROM continuous_experiments
                        WHERE experiment_name = %s
                    """, (new_name,))

                    if cursor.fetchone()['cnt'] == 0:
                        break  # Name is unique

                    # Name exists, increment and try again
                    run_number += 1
                    new_name = f"{name_base} ({run_number})"

            # Generate new experiment ID
            new_experiment_id = f"exp_{uuid.uuid4().hex[:8]}"

            # Calculate total cycles
            total_minutes = source_exp['target_duration_weeks'] * 7 * 24 * 60
            interval_minutes = source_exp['recording_interval_minutes']
            total_cycles = int(total_minutes / interval_minutes)

            # Create new experiment with same configuration
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
                 status, current_cycle, total_cycles_expected, start_time, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                new_experiment_id,
                new_name,
                source_exp['target_duration_weeks'],
                source_exp['recording_interval_minutes'],
                source_exp['playback_file'],
                source_exp['recording_duration_seconds'],
                source_exp['output_prefix'],
                source_exp['channel_1_substance'],
                source_exp['channel_2_substance'],
                source_exp['beaker_1_role'],
                source_exp['beaker_1_content'],
                source_exp['beaker_2_role'],
                source_exp['beaker_2_content'],
                source_exp['faraday_cage_used'],
                source_exp['researcher_name'],
                source_exp['auto_qc_enabled'],
                source_exp['auto_qc_min_separation_score'],
                source_exp['auto_qc_auto_approve_threshold'],
                source_exp['auto_qc_auto_reject_threshold'],
                source_exp['auto_qc_min_samples_per_channel'],
                source_exp['training_sliding_window_weeks'],
                source_exp['training_batch_size'],
                source_exp['training_epochs_per_cycle'],
                'stopped',  # Initial status
                0,  # current_cycle
                total_cycles
            ))
            conn.commit()

        logger.info(f"Duplicated experiment {experiment_id} as {new_experiment_id} ('{new_name}')")

        result = {
            'success': True,
            'experiment_id': new_experiment_id,
            'experiment_name': new_name,
            'message': f'Experiment duplicated successfully as "{new_name}"',
            'source_experiment_id': experiment_id
        }

        # Optionally start the experiment
        if auto_start:
            import subprocess
            import sys

            orchestrator_path = Path(__file__).parent.parent / 'src' / 'continuous' / 'recording_orchestrator.py'
            log_dir = Path(__file__).parent.parent / 'logs' / 'continuous'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{new_experiment_id}.log"

            # Start orchestrator as daemon process
            with open(log_file, 'a') as log_handle:
                process = subprocess.Popen(
                    [sys.executable, str(orchestrator_path), new_experiment_id, '--log-level', 'INFO'],
                    stdout=log_handle,
                    stderr=log_handle,
                    start_new_session=True
                )

            # Update status and PID
            with db.get_connection() as conn:
                cursor = conn.cursor(buffered=True)
                cursor.execute("""
                    UPDATE continuous_experiments
                    SET orchestrator_pid = %s,
                        status = 'running',
                        updated_at = NOW()
                    WHERE experiment_id = %s
                """, (process.pid, new_experiment_id))
                conn.commit()

            logger.info(f"Auto-started duplicated experiment {new_experiment_id} (PID: {process.pid})")
            result['started'] = True
            result['pid'] = process.pid

        return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to duplicate experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>', methods=['DELETE'])
def api_continuous_experiment_delete(experiment_id):
    """
    API endpoint: Delete a continuous experiment and all related data

    Deletes:
    - Experiment record from continuous_experiments
    - All related alerts from experiment_alerts
    - All related cycles from recording_cycles
    - All related recording sessions from recording_sessions
    - All extracted features from features table
    - Model files from disk

    Safety: Only allows deletion if experiment is not running
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Check if experiment exists and get status
            cursor.execute("""
                SELECT experiment_id, experiment_name, status, orchestrator_pid
                FROM continuous_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            experiment = cursor.fetchone()
            if not experiment:
                return jsonify({'success': False, 'error': 'Experiment not found'}), 404

            # Safety check: don't delete running experiments
            if experiment['status'] == 'running':
                return jsonify({
                    'success': False,
                    'error': 'Cannot delete running experiment. Stop or kill it first.'
                }), 400

            # Count related records before deletion
            cursor.execute("SELECT COUNT(*) as cnt FROM experiment_alerts WHERE experiment_id = %s", (experiment_id,))
            alert_count = cursor.fetchone()['cnt']

            cursor.execute("SELECT COUNT(*) as cnt FROM recording_cycles WHERE experiment_id = %s", (experiment_id,))
            cycle_count = cursor.fetchone()['cnt']

            # Get session IDs for this experiment to delete features and recording_sessions
            cursor.execute("""
                SELECT session_id FROM recording_cycles
                WHERE experiment_id = %s AND session_id IS NOT NULL
            """, (experiment_id,))
            session_ids = [row['session_id'] for row in cursor.fetchall()]

            # Count features and recording sessions
            features_count = 0
            sessions_count = 0
            if session_ids:
                placeholders = ','.join(['%s'] * len(session_ids))

                # Count features
                cursor.execute(f"""
                    SELECT COUNT(*) as cnt FROM features
                    WHERE session_id IN ({placeholders})
                """, session_ids)
                features_count = cursor.fetchone()['cnt']

                # Count recording sessions
                cursor.execute(f"""
                    SELECT COUNT(*) as cnt FROM recording_sessions
                    WHERE experiment_id = %s
                """, (experiment_id,))
                sessions_count = cursor.fetchone()['cnt']

            # Delete related records (cascade)
            logger.info(f"Deleting experiment {experiment_id}: {cycle_count} cycles, {alert_count} alerts, {sessions_count} sessions, {features_count} features")

            cursor.execute("DELETE FROM experiment_alerts WHERE experiment_id = %s", (experiment_id,))
            cursor.execute("DELETE FROM recording_cycles WHERE experiment_id = %s", (experiment_id,))

            # Delete features for this experiment's sessions
            if session_ids:
                placeholders = ','.join(['%s'] * len(session_ids))
                cursor.execute(f"DELETE FROM features WHERE session_id IN ({placeholders})", session_ids)

            # Delete recording sessions for this experiment
            cursor.execute("DELETE FROM recording_sessions WHERE experiment_id = %s", (experiment_id,))

            # Delete the experiment itself
            cursor.execute("DELETE FROM continuous_experiments WHERE experiment_id = %s", (experiment_id,))

            conn.commit()

            # Delete model files from disk
            model_files_deleted = 0
            try:
                import shutil
                model_dir = Path(f"models/continuous/{experiment_id}")
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    model_files_deleted = 1
                    logger.info(f"Deleted model directory: {model_dir}")
            except Exception as e:
                logger.warning(f"Failed to delete model directory: {e}")

            logger.info(f"Successfully deleted experiment {experiment_id} ({experiment['experiment_name']})")

            return jsonify({
                'success': True,
                'message': f'Experiment "{experiment["experiment_name"]}" deleted successfully',
                'deleted': {
                    'experiment': experiment['experiment_name'],
                    'cycles': cycle_count,
                    'alerts': alert_count,
                    'recording_sessions': sessions_count,
                    'features': features_count,
                    'model_files': model_files_deleted
                }
            })

    except Exception as e:
        logger.error(f"Failed to delete experiment {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/cycles', methods=['GET'])
def api_continuous_experiment_cycles(experiment_id):
    """
    API endpoint: Get cycle history for an experiment

    Query parameters:
        - limit: max results (default: 100)
        - offset: pagination offset (default: 0)
        - status: filter by cycle status
    """
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    status_filter = request.args.get('status', None)

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            query = """
                SELECT * FROM recording_cycles
                WHERE experiment_id = %s
            """
            params = [experiment_id]

            if status_filter:
                query += " AND status = %s"
                params.append(status_filter)

            query += " ORDER BY cycle_number DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            cycles = cursor.fetchall()

            # Convert datetime objects to strings
            for cycle in cycles:
                for key, value in cycle.items():
                    if isinstance(value, datetime):
                        cycle[key] = value.isoformat()

            return jsonify({
                'success': True,
                'cycles': cycles,
                'limit': limit,
                'offset': offset
            })

    except Exception as e:
        logger.error(f"Failed to fetch cycles for {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/stats', methods=['GET'])
def api_continuous_experiment_stats(experiment_id):
    """
    API endpoint: Get statistics for an experiment

    Returns:
        - QC pass rate
        - Average accuracy over time
        - Cycle completion rate
        - Alert counts by severity
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor(dictionary=True, buffered=True)

            # QC statistics
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN qc_passed = TRUE THEN 1 END) as qc_passes,
                    COUNT(CASE WHEN qc_passed = FALSE THEN 1 END) as qc_failures,
                    AVG(CASE WHEN qc_passed = TRUE THEN qc_separation_score END) as avg_separation_score
                FROM recording_cycles
                WHERE experiment_id = %s
                AND qc_passed IS NOT NULL
            """, (experiment_id,))
            qc_stats = cursor.fetchone()

            # Training statistics
            cursor.execute("""
                SELECT
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_cycles,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_cycles,
                    AVG(model_accuracy) as avg_accuracy,
                    MAX(model_accuracy) as max_accuracy,
                    AVG(training_time_seconds) as avg_training_time
                FROM recording_cycles
                WHERE experiment_id = %s
            """, (experiment_id,))
            training_stats = cursor.fetchone()

            # Alert statistics
            cursor.execute("""
                SELECT severity, COUNT(*) as count
                FROM experiment_alerts
                WHERE experiment_id = %s
                GROUP BY severity
            """, (experiment_id,))
            alert_stats = {row['severity']: row['count'] for row in cursor.fetchall()}

            # Accuracy over time (last 20 cycles)
            cursor.execute("""
                SELECT cycle_number, model_accuracy, end_time
                FROM recording_cycles
                WHERE experiment_id = %s
                AND model_accuracy IS NOT NULL
                ORDER BY cycle_number ASC
                LIMIT 50
            """, (experiment_id,))
            accuracy_history = cursor.fetchall()

            for row in accuracy_history:
                if row['end_time']:
                    row['end_time'] = row['end_time'].isoformat()

            return jsonify({
                'success': True,
                'qc': qc_stats,
                'training': training_stats,
                'alerts': alert_stats,
                'accuracy_history': accuracy_history
            })

    except Exception as e:
        logger.error(f"Failed to fetch stats for {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/continuous/experiments/<experiment_id>/visualizations', methods=['GET'])
def api_continuous_experiment_visualizations(experiment_id):
    """
    API endpoint: Get available visualizations for an experiment

    Scans the model/output directory for generated figures and returns them.

    Returns:
        - List of figure paths with captions
    """
    try:
        # Look for figures in multiple possible locations
        base_paths = [
            Path(f"models/continuous/{experiment_id}"),
            Path(f"reports/continuous/{experiment_id}"),
            Path(f"data/continuous/{experiment_id}/plots")
        ]

        figures = []

        for base_path in base_paths:
            if not base_path.exists():
                continue

            # Find all PNG/JPG files recursively
            for img_path in base_path.rglob("*.png"):
                # Convert to absolute path first, then make relative to cwd
                abs_path = img_path.resolve()
                try:
                    relative_path = abs_path.relative_to(Path.cwd().resolve())
                    figures.append({
                        'path': f'/{relative_path}',
                        'caption': generate_caption_from_filename(img_path.name),
                        'filename': img_path.name,
                        'size': img_path.stat().st_size,
                        'modified': datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
                    })
                except ValueError:
                    # Path is not relative to cwd, skip it
                    logger.warning(f"Skipping file outside project: {abs_path}")
                    continue

            for img_path in base_path.rglob("*.jpg"):
                abs_path = img_path.resolve()
                try:
                    relative_path = abs_path.relative_to(Path.cwd().resolve())
                    figures.append({
                        'path': f'/{relative_path}',
                        'caption': generate_caption_from_filename(img_path.name),
                        'filename': img_path.name,
                        'size': img_path.stat().st_size,
                        'modified': datetime.fromtimestamp(img_path.stat().st_mtime).isoformat()
                    })
                except ValueError:
                    logger.warning(f"Skipping file outside project: {abs_path}")
                    continue

        # Sort by modification time (newest first)
        figures.sort(key=lambda x: x['modified'], reverse=True)

        logger.info(f"Found {len(figures)} visualizations for {experiment_id}")

        return jsonify({
            'success': True,
            'figures': figures,
            'count': len(figures)
        })

    except Exception as e:
        logger.error(f"Failed to fetch visualizations for {experiment_id}: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def generate_caption_from_filename(filename):
    """
    Generate a human-readable caption from a filename.

    Examples:
        temporal_umap_12weeks.png -> Temporal UMAP (12 weeks)
        roc_progression.png -> ROC Progression
        accuracy_vs_time.png -> Accuracy vs Time
    """
    # Remove extension
    name = filename.rsplit('.', 1)[0]

    # Replace underscores and hyphens with spaces
    name = name.replace('_', ' ').replace('-', ' ')

    # Capitalize words
    words = name.split()
    capitalized = []
    for word in words:
        # Handle special cases
        if word.lower() in ['umap', 'tsne', 'pca', 'roc', 'pr', 'qc', 'vs']:
            capitalized.append(word.upper())
        elif word.isdigit():
            capitalized.append(word)
        elif 'week' in word.lower():
            capitalized.append(word.lower())
        else:
            capitalized.append(word.capitalize())

    return ' '.join(capitalized)


# ========================================================================
# STATIC FILE SERVING FOR VISUALIZATIONS
# ========================================================================

@app.route('/models/<path:filename>')
def serve_model_files(filename):
    """Serve files from models directory"""
    return send_from_directory(Path.cwd() / 'models', filename)


@app.route('/reports/<path:filename>')
def serve_report_files(filename):
    """Serve files from reports directory"""
    return send_from_directory(Path.cwd() / 'reports', filename)


@app.route('/data/<path:filename>')
def serve_data_files(filename):
    """Serve files from data directory (plots only, for security)"""
    # Security: only allow access to plot files
    if 'plots' not in filename:
        return jsonify({'error': 'Access denied'}), 403
    return send_from_directory(Path.cwd() / 'data', filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Recording Management Web Interface')
    parser.add_argument('--port', type=int, default=5001, help='Port to run on (default: 5001)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # Run stale experiment cleanup on startup (handles system restarts/hibernation)
    cleanup_stale_experiments_on_startup()

    logger.info(f"Starting Recording Management Web Interface on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
