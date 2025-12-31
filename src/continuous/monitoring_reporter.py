"""
Monitoring and Reporting System for Continuous Learning

Tracks system health, performance metrics, and sends automated reports.

Features:
- Database health monitoring (size, growth rate, distribution)
- Model performance tracking (accuracy trends, drift detection)
- Ingestion metrics (throughput, errors, data quality)
- Weekly email reports
- Alert system for anomalies
- Performance visualization exports

Usage:
    from continuous.monitoring_reporter import SystemMonitor

    monitor = SystemMonitor(config, db_path, email_config)
    report = monitor.generate_weekly_report()
    monitor.send_email_report(report, recipients)
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import json

from .feature_database import FeatureDatabase

logger = logging.getLogger(__name__)


class SystemHealthMetrics:
    """Calculate system health metrics"""

    @staticmethod
    def calculate_database_health(db: FeatureDatabase) -> Dict:
        """
        Calculate database health metrics.

        Returns:
            Dict with database health indicators
        """
        stats = db.get_database_stats()

        # Calculate class balance
        total = stats['total_samples']
        class_0 = stats.get('class_0_count', 0)
        class_1 = stats.get('class_1_count', 0)

        balance_ratio = min(class_0, class_1) / max(class_0, class_1) if max(class_0, class_1) > 0 else 0
        is_balanced = balance_ratio > 0.7  # Consider balanced if >70% ratio

        # Storage metrics
        db_size_gb = stats['db_size_mb'] / 1024
        capacity_2tb = 2048  # GB
        utilization_pct = (db_size_gb / capacity_2tb) * 100

        return {
            'total_samples': total,
            'class_0_count': class_0,
            'class_1_count': class_1,
            'balance_ratio': balance_ratio,
            'is_balanced': is_balanced,
            'db_size_gb': db_size_gb,
            'storage_utilization_pct': utilization_pct,
            'estimated_days_remaining': (capacity_2tb - db_size_gb) / (1.13 / 365) if db_size_gb > 0 else float('inf')
        }

    @staticmethod
    def calculate_growth_rate(db: FeatureDatabase, days: int = 7) -> Dict:
        """
        Calculate data growth rate.

        Args:
            db: Feature database
            days: Number of days to analyze

        Returns:
            Dict with growth metrics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get sample counts by date (would need to query by date ranges)
        # For now, return estimated metrics
        stats = db.get_database_stats()

        return {
            'samples_per_day': stats['total_samples'] / max(days, 1),
            'growth_period_days': days,
            'projected_monthly_samples': (stats['total_samples'] / max(days, 1)) * 30
        }

    @staticmethod
    def detect_data_drift(db: FeatureDatabase, weeks: int = 4) -> Dict:
        """
        Detect data distribution drift.

        Compares recent data to historical baseline.

        Args:
            db: Feature database
            weeks: Weeks of recent data to analyze

        Returns:
            Dict with drift indicators
        """
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)

        X_recent, y_recent = db.load_features_by_date_range(start_date, end_date, max_samples=1000)

        # Load baseline (older data)
        baseline_end = start_date
        baseline_start = baseline_end - timedelta(weeks=weeks)

        X_baseline, y_baseline = db.load_features_by_date_range(
            baseline_start, baseline_end, max_samples=1000
        )

        if len(X_recent) < 50 or len(X_baseline) < 50:
            return {
                'drift_detected': False,
                'reason': 'insufficient_data',
                'recent_samples': len(X_recent),
                'baseline_samples': len(X_baseline)
            }

        # Calculate distribution statistics
        recent_mean = X_recent.mean(axis=0)
        baseline_mean = X_baseline.mean(axis=0)
        recent_std = X_recent.std(axis=0)
        baseline_std = X_baseline.std(axis=0)

        # Mean shift (normalized by baseline std)
        mean_shift = np.abs(recent_mean - baseline_mean) / (baseline_std + 1e-8)
        max_mean_shift = np.max(mean_shift)
        avg_mean_shift = np.mean(mean_shift)

        # Std shift
        std_ratio = recent_std / (baseline_std + 1e-8)
        max_std_ratio = np.max(std_ratio)

        # Drift thresholds
        drift_detected = (max_mean_shift > 3.0) or (max_std_ratio > 2.0 or max_std_ratio < 0.5)

        return {
            'drift_detected': drift_detected,
            'max_mean_shift': float(max_mean_shift),
            'avg_mean_shift': float(avg_mean_shift),
            'max_std_ratio': float(max_std_ratio),
            'recent_samples': len(X_recent),
            'baseline_samples': len(X_baseline),
            'drift_severity': 'high' if max_mean_shift > 5 else 'medium' if max_mean_shift > 3 else 'low'
        }


class PerformanceTracker:
    """Track model performance over time"""

    @staticmethod
    def get_training_history(db: FeatureDatabase, n_runs: int = 10) -> List[Dict]:
        """
        Get recent training run history.

        Args:
            db: Feature database
            n_runs: Number of recent runs to retrieve

        Returns:
            List of training run records
        """
        # Query database for recent training runs
        import sqlite3

        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT run_timestamp, model_path, train_accuracy, val_accuracy,
                   test_accuracy, roc_auc, update_mode, n_training_samples
            FROM training_runs
            ORDER BY run_timestamp DESC
            LIMIT ?
        ''', (n_runs,))

        runs = []
        for row in cursor.fetchall():
            runs.append({
                'timestamp': row[0],
                'model_path': row[1],
                'train_accuracy': row[2],
                'val_accuracy': row[3],
                'test_accuracy': row[4],
                'roc_auc': row[5],
                'update_mode': row[6],
                'n_training_samples': row[7]
            })

        conn.close()

        return runs

    @staticmethod
    def calculate_performance_trend(training_history: List[Dict]) -> Dict:
        """
        Calculate performance trend from training history.

        Args:
            training_history: List of training run records

        Returns:
            Dict with trend metrics
        """
        if len(training_history) < 2:
            return {
                'trend': 'insufficient_data',
                'n_runs': len(training_history)
            }

        # Extract metrics
        accuracies = [run['test_accuracy'] for run in training_history if run['test_accuracy'] is not None]

        if len(accuracies) < 2:
            return {'trend': 'insufficient_data'}

        # Calculate trend
        recent_avg = np.mean(accuracies[:3]) if len(accuracies) >= 3 else accuracies[0]
        older_avg = np.mean(accuracies[-3:]) if len(accuracies) >= 3 else accuracies[-1]

        improvement = recent_avg - older_avg
        trend = 'improving' if improvement > 0.02 else 'declining' if improvement < -0.02 else 'stable'

        return {
            'trend': trend,
            'recent_avg_accuracy': float(recent_avg),
            'older_avg_accuracy': float(older_avg),
            'improvement': float(improvement),
            'best_accuracy': float(max(accuracies)),
            'worst_accuracy': float(min(accuracies)),
            'current_accuracy': float(accuracies[0]),
            'n_runs': len(accuracies)
        }


class WeeklyReport:
    """Generate comprehensive weekly report"""

    def __init__(self, db: FeatureDatabase):
        self.db = db
        self.report_time = datetime.now()

    def generate(self) -> Dict:
        """
        Generate complete weekly report.

        Returns:
            Dict with all report sections
        """
        logger.info("Generating weekly report...")

        report = {
            'report_time': self.report_time.isoformat(),
            'report_period': f"{(self.report_time - timedelta(days=7)).strftime('%Y-%m-%d')} to {self.report_time.strftime('%Y-%m-%d')}",
            'database_health': SystemHealthMetrics.calculate_database_health(self.db),
            'growth_metrics': SystemHealthMetrics.calculate_growth_rate(self.db, days=7),
            'drift_analysis': SystemHealthMetrics.detect_data_drift(self.db, weeks=4),
            'training_history': PerformanceTracker.get_training_history(self.db, n_runs=10),
            'performance_trend': None  # Will be calculated from history
        }

        # Calculate performance trend
        if report['training_history']:
            report['performance_trend'] = PerformanceTracker.calculate_performance_trend(
                report['training_history']
            )

        logger.info("Weekly report generated")

        return report

    def format_as_text(self, report: Dict) -> str:
        """
        Format report as plain text for email.

        Args:
            report: Report dictionary

        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("CONTINUOUS LEARNING WEEKLY REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {report['report_time']}")
        lines.append(f"Period: {report['report_period']}")
        lines.append("")

        # Database Health
        lines.append("DATABASE HEALTH")
        lines.append("-" * 70)
        db_health = report['database_health']
        lines.append(f"  Total Samples: {db_health['total_samples']:,}")
        lines.append(f"  Class 0: {db_health['class_0_count']:,}  |  Class 1: {db_health['class_1_count']:,}")
        lines.append(f"  Balance Ratio: {db_health['balance_ratio']:.2f}  {'✓ BALANCED' if db_health['is_balanced'] else '⚠ IMBALANCED'}")
        lines.append(f"  Database Size: {db_health['db_size_gb']:.2f} GB ({db_health['storage_utilization_pct']:.1f}% of 2TB)")
        lines.append(f"  Est. Days Until Full: {db_health['estimated_days_remaining']:.0f}")
        lines.append("")

        # Growth Metrics
        lines.append("GROWTH METRICS")
        lines.append("-" * 70)
        growth = report['growth_metrics']
        lines.append(f"  Samples/Day: {growth['samples_per_day']:.0f}")
        lines.append(f"  Projected Monthly: {growth['projected_monthly_samples']:.0f} samples")
        lines.append("")

        # Data Drift
        lines.append("DATA DRIFT ANALYSIS")
        lines.append("-" * 70)
        drift = report['drift_analysis']
        if drift.get('drift_detected'):
            lines.append(f"  ⚠ DRIFT DETECTED ({drift.get('drift_severity', 'unknown')} severity)")
            lines.append(f"  Max Mean Shift: {drift.get('max_mean_shift', 0):.2f}σ")
            lines.append(f"  Avg Mean Shift: {drift.get('avg_mean_shift', 0):.2f}σ")
        else:
            lines.append(f"  ✓ No significant drift detected")
            if drift.get('reason') == 'insufficient_data':
                lines.append(f"  (Insufficient data: {drift.get('recent_samples', 0)} recent samples)")
        lines.append("")

        # Performance Trend
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 70)
        perf = report.get('performance_trend')
        if perf and perf.get('trend') != 'insufficient_data':
            lines.append(f"  Trend: {perf['trend'].upper()}")
            lines.append(f"  Current Accuracy: {perf['current_accuracy']:.4f}")
            lines.append(f"  Best Accuracy: {perf['best_accuracy']:.4f}")
            lines.append(f"  Recent Avg: {perf['recent_avg_accuracy']:.4f}")
            lines.append(f"  Change: {perf['improvement']:+.4f}")
        else:
            lines.append(f"  No training runs in reporting period")
        lines.append("")

        # Recent Training Runs
        history = report.get('training_history', [])
        if history:
            lines.append("RECENT TRAINING RUNS")
            lines.append("-" * 70)
            for i, run in enumerate(history[:5], 1):
                lines.append(f"  {i}. {run.get('timestamp', 'N/A')}")
                lines.append(f"     Accuracy: {run.get('test_accuracy', 0):.4f}  |  AUC: {run.get('roc_auc', 0):.4f}")
                lines.append(f"     Mode: {run.get('update_mode', 'N/A')}  |  Samples: {run.get('n_training_samples', 0):,}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("End of Report")
        lines.append("=" * 70)

        return "\n".join(lines)


class SystemMonitor:
    """
    System monitor for continuous learning pipeline.

    Orchestrates monitoring, reporting, and alerting.
    """

    def __init__(self,
                 db_path: Path,
                 email_config: Optional[Dict] = None):
        """
        Initialize system monitor.

        Args:
            db_path: Path to feature database
            email_config: Optional email configuration dict with keys:
                - smtp_server: SMTP server address
                - smtp_port: SMTP port (default: 587)
                - sender_email: From address
                - sender_password: SMTP password
                - use_tls: Whether to use TLS (default: True)
        """
        self.db = FeatureDatabase(db_path)
        self.email_config = email_config or {}

        logger.info("SystemMonitor initialized")

    def generate_weekly_report(self) -> Dict:
        """Generate weekly report"""
        reporter = WeeklyReport(self.db)
        return reporter.generate()

    def format_report_as_text(self, report: Dict) -> str:
        """Format report as text"""
        reporter = WeeklyReport(self.db)
        return reporter.format_as_text(report)

    def send_email_report(self,
                         report: Dict,
                         recipients: List[str],
                         subject: Optional[str] = None) -> bool:
        """
        Send email report.

        Args:
            report: Report dictionary
            recipients: List of recipient email addresses
            subject: Optional custom subject line

        Returns:
            True if email sent successfully
        """
        if not self.email_config.get('smtp_server'):
            logger.warning("Email config not provided, skipping email send")
            return False

        try:
            # Format report
            reporter = WeeklyReport(self.db)
            body = reporter.format_as_text(report)

            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config.get('sender_email', 'noreply@example.com')
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject or f"Continuous Learning Weekly Report - {datetime.now().strftime('%Y-%m-%d')}"

            # Attach body
            msg.attach(MIMEText(body, 'plain'))

            # Send email
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config.get('smtp_port', 587)
            use_tls = self.email_config.get('use_tls', True)

            server = smtplib.SMTP(smtp_server, smtp_port)

            if use_tls:
                server.starttls()

            if self.email_config.get('sender_password'):
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )

            server.send_message(msg)
            server.quit()

            logger.info(f"Email report sent to {len(recipients)} recipient(s)")
            return True

        except Exception as e:
            logger.error(f"Failed to send email report: {e}", exc_info=True)
            return False

    def check_alerts(self, report: Dict) -> List[Dict]:
        """
        Check for alert conditions.

        Args:
            report: Weekly report

        Returns:
            List of alerts
        """
        alerts = []

        # Check storage utilization
        db_health = report.get('database_health', {})
        if db_health.get('storage_utilization_pct', 0) > 80:
            alerts.append({
                'severity': 'warning',
                'type': 'storage',
                'message': f"Storage {db_health['storage_utilization_pct']:.1f}% full"
            })

        # Check class balance
        if not db_health.get('is_balanced', True):
            alerts.append({
                'severity': 'warning',
                'type': 'class_balance',
                'message': f"Class imbalance detected (ratio: {db_health.get('balance_ratio', 0):.2f})"
            })

        # Check data drift
        drift = report.get('drift_analysis', {})
        if drift.get('drift_detected'):
            alerts.append({
                'severity': 'high' if drift.get('drift_severity') == 'high' else 'medium',
                'type': 'data_drift',
                'message': f"Data drift detected ({drift.get('drift_severity')} severity)"
            })

        # Check performance
        perf = report.get('performance_trend', {})
        if perf.get('trend') == 'declining':
            alerts.append({
                'severity': 'medium',
                'type': 'performance',
                'message': f"Model performance declining (change: {perf.get('improvement', 0):.4f})"
            })

        return alerts
