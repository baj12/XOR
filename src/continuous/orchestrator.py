"""
Autonomous Orchestration for Continuous Learning System

Coordinates all components for 24/7 autonomous operation:
- Continuous data ingestion from stereo WAV streams
- Weekly model training updates
- Performance monitoring and drift detection
- Weekly report generation and email alerts
- Automatic recovery from failures

This script is designed to run indefinitely as a daemon process.

Usage:
    # Basic usage (runs forever)
    python orchestrator.py --config config.yaml --db features.db

    # Testing mode (runs for specified duration)
    python orchestrator.py --config config.yaml --db features.db --test-hours 24

    # Dry run (no actual training or emails)
    python orchestrator.py --config config.yaml --db features.db --dry-run

Configuration:
    orchestration:
      data_check_interval_minutes: 60    # How often to check for new data
      training_day_of_week: 0            # 0=Monday, 6=Sunday
      training_hour: 2                   # Hour to run training (0-23)
      email_recipients:
        - user@example.com
      email_smtp_server: smtp.gmail.com
      email_smtp_port: 587
      min_samples_for_training: 1000     # Minimum samples before training
      alert_on_drift: true
      alert_on_performance_drop: true
      performance_drop_threshold: 0.05   # Alert if accuracy drops >5%
"""

import logging
import time
import smtplib
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import schedule

from .continuous_ingestion import ContinuousIngestionPipeline
from .incremental_trainer import IncrementalTrainer
from .monitoring_reporter import SystemMonitor, WeeklyReport
from .feature_database import FeatureDatabase

logger = logging.getLogger(__name__)


class EmailReporter:
    """Handle email notifications and weekly reports"""

    def __init__(self, config):
        """
        Initialize email reporter.

        Args:
            config: Configuration with email settings
        """
        self.config = config
        self.smtp_server = getattr(config.orchestration, 'email_smtp_server', 'smtp.gmail.com')
        self.smtp_port = getattr(config.orchestration, 'email_smtp_port', 587)
        self.sender = getattr(config.orchestration, 'email_sender', 'noreply@example.com')
        self.password = getattr(config.orchestration, 'email_password', None)
        self.recipients = getattr(config.orchestration, 'email_recipients', [])

        logger.info(f"EmailReporter initialized: {len(self.recipients)} recipients")

    def send_email(self,
                   subject: str,
                   body: str,
                   attachments: Optional[List[Path]] = None,
                   html: bool = False) -> bool:
        """
        Send email with optional attachments.

        Args:
            subject: Email subject
            body: Email body (plain text or HTML)
            attachments: Optional list of file paths to attach
            html: Whether body is HTML (default: plain text)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.recipients:
            logger.warning("No email recipients configured, skipping email")
            return False

        if not self.password:
            logger.warning("No email password configured, skipping email")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = subject
            msg['Date'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')

            # Attach body
            mime_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, mime_type))

            # Attach files
            if attachments:
                for file_path in attachments:
                    if not file_path.exists():
                        logger.warning(f"Attachment not found: {file_path}")
                        continue

                    with open(file_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())

                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', f'attachment; filename={file_path.name}')
                    msg.attach(part)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.send_message(msg)

            logger.info(f"Email sent: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}", exc_info=True)
            return False

    def send_weekly_report(self, report_path: Path) -> bool:
        """
        Send weekly performance report.

        Args:
            report_path: Path to weekly report markdown file

        Returns:
            True if sent successfully
        """
        if not report_path.exists():
            logger.error(f"Report not found: {report_path}")
            return False

        # Read report
        with open(report_path, 'r') as f:
            report_content = f.read()

        subject = f"Continuous Learning Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"

        # Convert markdown to simple HTML
        html_body = self._markdown_to_html(report_content)

        return self.send_email(subject, html_body, html=True)

    def send_alert(self, alert_type: str, details: Dict) -> bool:
        """
        Send alert email for critical issues.

        Args:
            alert_type: Type of alert (drift, performance, storage, error)
            details: Alert details dictionary

        Returns:
            True if sent successfully
        """
        subject = f"⚠️ Continuous Learning Alert: {alert_type.upper()}"

        body = f"""
ALERT: {alert_type}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Details:
{self._format_dict(details)}

This is an automated alert from the continuous learning system.
Please review the system status and take appropriate action if needed.
"""

        return self.send_email(subject, body)

    @staticmethod
    def _markdown_to_html(markdown: str) -> str:
        """Simple markdown to HTML conversion"""
        html = markdown.replace('\n\n', '</p><p>')
        html = html.replace('\n', '<br>')
        html = html.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')
        html = f"<html><body><p>{html}</p></body></html>"
        return html

    @staticmethod
    def _format_dict(d: Dict) -> str:
        """Format dictionary for email body"""
        return '\n'.join(f"  {k}: {v}" for k, v in d.items())


class ContinuousLearningOrchestrator:
    """
    Orchestrates autonomous continuous learning system.

    Coordinates:
    - Continuous data ingestion
    - Periodic model training
    - Performance monitoring
    - Alerting and reporting
    """

    def __init__(self,
                 config,
                 db_path: Path,
                 model_dir: Path,
                 report_dir: Path,
                 data_dir: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            config: Configuration object
            db_path: Path to feature database
            model_dir: Directory for saving models
            report_dir: Directory for saving reports
            data_dir: Optional directory to monitor for new WAV files
        """
        self.config = config
        self.db_path = Path(db_path)
        self.model_dir = Path(model_dir)
        self.report_dir = Path(report_dir)
        self.data_dir = Path(data_dir) if data_dir else None

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.db = FeatureDatabase(self.db_path)
        self.ingestion = ContinuousIngestionPipeline(config, self.db_path) if data_dir else None
        self.trainer = IncrementalTrainer(config, self.db_path, self.model_dir)
        self.monitor = SystemMonitor(self.db_path)
        self.email = EmailReporter(config)

        # State tracking
        self.last_training_time = None
        self.last_report_time = None
        self.is_running = False
        self.error_count = 0
        self.max_consecutive_errors = 10

        # Get configuration
        self.check_interval_minutes = getattr(config.orchestration, 'data_check_interval_minutes', 60)
        self.training_day_of_week = getattr(config.orchestration, 'training_day_of_week', 0)  # Monday
        self.training_hour = getattr(config.orchestration, 'training_hour', 2)  # 2 AM
        self.min_samples_for_training = getattr(config.orchestration, 'min_samples_for_training', 1000)
        self.alert_on_drift = getattr(config.orchestration, 'alert_on_drift', True)
        self.alert_on_performance = getattr(config.orchestration, 'alert_on_performance_drop', True)
        self.performance_threshold = getattr(config.orchestration, 'performance_drop_threshold', 0.05)

        logger.info("ContinuousLearningOrchestrator initialized")
        logger.info(f"Database: {self.db_path}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Report directory: {self.report_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Check interval: {self.check_interval_minutes} minutes")
        logger.info(f"Training schedule: Day {self.training_day_of_week} at {self.training_hour}:00")

    def check_and_ingest_data(self) -> Dict:
        """
        Check for new data and ingest if available.

        Returns:
            Dict with ingestion results
        """
        if not self.ingestion or not self.data_dir:
            return {'status': 'skipped', 'reason': 'no_data_directory'}

        logger.info("Checking for new data...")

        try:
            # Find new WAV files
            wav_files = sorted(self.data_dir.glob('*.wav'))

            if not wav_files:
                logger.info("No new data files found")
                return {'status': 'no_data', 'files_found': 0}

            # Ingest files
            results = self.ingestion.ingest_batch(wav_files)

            # Count successes
            n_success = sum(1 for r in results if r['status'] == 'success')
            n_failed = len(results) - n_success

            logger.info(f"Ingestion complete: {n_success} succeeded, {n_failed} failed")

            return {
                'status': 'success',
                'files_processed': len(results),
                'files_succeeded': n_success,
                'files_failed': n_failed
            }

        except Exception as e:
            logger.error(f"Data ingestion failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    def should_train_now(self) -> bool:
        """
        Check if it's time for scheduled training.

        Returns:
            True if training should run now
        """
        now = datetime.now()

        # Check if correct day of week
        if now.weekday() != self.training_day_of_week:
            return False

        # Check if correct hour
        if now.hour != self.training_hour:
            return False

        # Check if already trained today
        if self.last_training_time:
            time_since_training = now - self.last_training_time
            if time_since_training < timedelta(hours=23):
                return False

        # Check if enough data
        stats = self.db.get_database_stats()
        if stats['total_samples'] < self.min_samples_for_training:
            logger.info(f"Insufficient samples for training: {stats['total_samples']} < {self.min_samples_for_training}")
            return False

        return True

    def run_training(self) -> Dict:
        """
        Execute model training.

        Returns:
            Dict with training results
        """
        logger.info("=== Starting scheduled training ===")

        try:
            # Decide on training mode
            if self.last_training_time is None:
                # First training - full training
                logger.info("First training - performing full training")
                result = self.trainer.train_full(weeks=4, epochs=20, batch_size=32)
            else:
                # Incremental training
                logger.info("Incremental training with new data")
                result = self.trainer.train_incremental(weeks=1, epochs=5, batch_size=32)

            self.last_training_time = datetime.now()

            # Check performance
            if result['status'] == 'success':
                logger.info(f"Training complete: accuracy={result.get('test_accuracy', 'N/A'):.4f}")

                # Alert on performance drop
                if self.alert_on_performance and 'test_accuracy' in result:
                    if result['test_accuracy'] < (1.0 - self.performance_threshold):
                        self.email.send_alert('performance', {
                            'test_accuracy': result['test_accuracy'],
                            'threshold': 1.0 - self.performance_threshold,
                            'message': 'Model accuracy below threshold'
                        })

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    def run_monitoring(self) -> Dict:
        """
        Run monitoring checks and generate alerts if needed.

        Returns:
            Dict with monitoring results
        """
        logger.info("Running monitoring checks...")

        try:
            from .monitoring_reporter import SystemHealthMetrics

            results = {}

            # Check database health (static method)
            health = SystemHealthMetrics.calculate_database_health(self.db)
            results['health'] = health

            # Alert on storage
            if health['storage_utilization_pct'] > 90:
                self.email.send_alert('storage', {
                    'utilization': f"{health['storage_utilization_pct']:.1f}%",
                    'estimated_days_remaining': health.get('estimated_days_remaining', 'unknown')
                })

            # Alert on class imbalance
            if not health['is_balanced']:
                self.email.send_alert('class_imbalance', {
                    'balance_ratio': health['balance_ratio'],
                    'class_0_samples': health.get('class_0_samples', 'unknown'),
                    'class_1_samples': health.get('class_1_samples', 'unknown')
                })

            # Check for drift (static method)
            drift = SystemHealthMetrics.detect_data_drift(self.db, weeks=4)
            results['drift'] = drift

            if self.alert_on_drift and drift['drift_detected']:
                self.email.send_alert('drift', {
                    'drift_severity': drift['drift_severity'],
                    'max_mean_shift': drift.get('max_mean_shift', 'unknown'),
                    'max_std_ratio': drift.get('max_std_ratio', 'unknown')
                })

            logger.info(f"Monitoring complete: health OK={health.get('is_balanced', False)}, drift={drift['drift_detected']}")

            return results

        except Exception as e:
            logger.error(f"Monitoring failed: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}

    def generate_weekly_report(self) -> Optional[Path]:
        """
        Generate and send weekly report.

        Returns:
            Path to generated report or None if failed
        """
        logger.info("Generating weekly report...")

        try:
            # Generate report
            report = WeeklyReport(self.config, self.db_path, self.model_dir, self.report_dir)
            report_path = report.generate_report()

            # Send via email
            if report_path:
                self.email.send_weekly_report(report_path)
                self.last_report_time = datetime.now()
                logger.info(f"Weekly report sent: {report_path}")
                return report_path
            else:
                logger.warning("Failed to generate report")
                return None

        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            return None

    def run_cycle(self) -> None:
        """Execute one cycle of the orchestration loop"""
        try:
            logger.info(f"=== Orchestration Cycle: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

            # 1. Check and ingest new data
            if self.data_dir:
                ingestion_result = self.check_and_ingest_data()
                logger.info(f"Ingestion: {ingestion_result}")

            # 2. Check if training should run
            if self.should_train_now():
                training_result = self.run_training()
                logger.info(f"Training: {training_result}")

            # 3. Run monitoring checks
            monitoring_result = self.run_monitoring()
            logger.info(f"Monitoring: {monitoring_result}")

            # 4. Generate weekly report (if it's Monday)
            now = datetime.now()
            if now.weekday() == 0 and now.hour == 8:  # Monday 8 AM
                if not self.last_report_time or (now - self.last_report_time) > timedelta(days=6):
                    self.generate_weekly_report()

            # Reset error count on success
            self.error_count = 0

        except Exception as e:
            logger.error(f"Orchestration cycle failed: {e}", exc_info=True)
            self.error_count += 1

            # Send alert on repeated failures
            if self.error_count >= 3:
                self.email.send_alert('system_error', {
                    'error_count': self.error_count,
                    'last_error': str(e),
                    'traceback': traceback.format_exc()
                })

            # Stop if too many errors
            if self.error_count >= self.max_consecutive_errors:
                logger.critical(f"Too many consecutive errors ({self.error_count}), stopping orchestrator")
                self.is_running = False

    def run(self, test_duration_hours: Optional[int] = None, dry_run: bool = False) -> None:
        """
        Run orchestrator indefinitely (or for test duration).

        Args:
            test_duration_hours: Optional duration for testing (None = run forever)
            dry_run: If True, skip actual training and email sending
        """
        logger.info("Starting continuous learning orchestrator...")
        logger.info(f"Test duration: {test_duration_hours} hours" if test_duration_hours else "Running indefinitely")
        logger.info(f"Dry run: {dry_run}")

        self.is_running = True
        start_time = datetime.now()

        # Schedule tasks
        schedule.every(self.check_interval_minutes).minutes.do(self.run_cycle)

        # Initial cycle
        self.run_cycle()

        # Main loop
        while self.is_running:
            # Check test duration
            if test_duration_hours:
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                if elapsed >= test_duration_hours:
                    logger.info(f"Test duration reached ({test_duration_hours} hours), stopping")
                    break

            # Run scheduled tasks
            schedule.run_pending()

            # Sleep
            time.sleep(60)  # Check every minute

        logger.info("Orchestrator stopped")

    def stop(self) -> None:
        """Stop the orchestrator gracefully"""
        logger.info("Stopping orchestrator...")
        self.is_running = False


def main():
    """Main entry point for orchestrator"""
    import argparse
    import yaml
    from ..utils import load_config

    parser = argparse.ArgumentParser(description='Continuous Learning Orchestrator')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--db', type=str, required=True, help='Path to feature database')
    parser.add_argument('--model-dir', type=str, default='models', help='Model directory')
    parser.add_argument('--report-dir', type=str, default='reports', help='Report directory')
    parser.add_argument('--data-dir', type=str, help='Data directory to monitor (optional)')
    parser.add_argument('--test-hours', type=int, help='Test duration in hours (optional)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no training/emails)')
    parser.add_argument('--log', type=str, default='INFO', help='Log level')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = load_config(args.config)

    # Create orchestrator
    orchestrator = ContinuousLearningOrchestrator(
        config=config,
        db_path=Path(args.db),
        model_dir=Path(args.model_dir),
        report_dir=Path(args.report_dir),
        data_dir=Path(args.data_dir) if args.data_dir else None
    )

    # Run
    try:
        orchestrator.run(test_duration_hours=args.test_hours, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
        orchestrator.stop()


if __name__ == '__main__':
    main()
