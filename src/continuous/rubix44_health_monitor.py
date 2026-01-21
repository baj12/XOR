"""
Rubix44 Server Health Monitoring

Provides utilities for monitoring rubix44 server health, detecting crashes,
and implementing resilient retry logic.
"""

import time
import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


class Rubix44HealthMonitor:
    """Monitor rubix44 server health and handle crash recovery."""

    def __init__(self, api_url: str, logger: Optional[logging.Logger] = None):
        """
        Initialize health monitor.

        Args:
            api_url: Base URL of rubix44 API (e.g., http://10.0.0.58:5000)
            logger: Optional logger instance
        """
        self.api_url = api_url.rstrip('/')
        self.logger = logger or logging.getLogger(__name__)
        self._last_health_check: Optional[datetime] = None
        self._consecutive_failures = 0
        self._server_down_since: Optional[datetime] = None

    def is_healthy(self, timeout: int = 5) -> bool:
        """
        Check if server is healthy and responding.

        Args:
            timeout: Request timeout in seconds

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            # Try system health endpoint first (more comprehensive)
            response = requests.get(
                f"{self.api_url}/api/v1/system/health",
                timeout=timeout
            )

            if response.status_code == 200:
                data = response.json()
                self._consecutive_failures = 0
                self._server_down_since = None
                self._last_health_check = datetime.now()

                # Log crash info if available
                crashes = data.get('crash_history', {})
                if crashes.get('total_crashes', 0) > 0:
                    self.logger.warning(
                        f"Server has {crashes['total_crashes']} crashes in history. "
                        f"Uptime: {data.get('uptime_seconds', 0):.1f}s"
                    )

                return True

        except requests.exceptions.RequestException:
            # Fall back to basic health endpoint
            try:
                response = requests.get(
                    f"{self.api_url}/api/v1/health",
                    timeout=timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        self._consecutive_failures = 0
                        self._server_down_since = None
                        self._last_health_check = datetime.now()
                        return True
            except requests.exceptions.RequestException:
                pass

        # Health check failed
        self._consecutive_failures += 1
        if self._server_down_since is None:
            self._server_down_since = datetime.now()
            self.logger.error(f"Rubix44 server became unresponsive at {self._server_down_since}")

        return False

    def wait_for_recovery(
        self,
        max_wait_seconds: int = 600,
        check_interval: int = 10
    ) -> bool:
        """
        Wait for server to recover from crash/downtime.

        Args:
            max_wait_seconds: Maximum time to wait for recovery
            check_interval: How often to check (seconds)

        Returns:
            True if server recovered, False if timeout
        """
        start_time = time.time()
        self.logger.info(
            f"Waiting for rubix44 server to recover "
            f"(max {max_wait_seconds}s, checking every {check_interval}s)..."
        )

        while time.time() - start_time < max_wait_seconds:
            if self.is_healthy():
                downtime = time.time() - (
                    self._server_down_since.timestamp()
                    if self._server_down_since
                    else start_time
                )
                self.logger.info(
                    f"✅ Server recovered after {downtime:.1f}s downtime"
                )
                return True

            elapsed = time.time() - start_time
            self.logger.debug(
                f"Server still down... ({elapsed:.0f}s / {max_wait_seconds}s)"
            )
            time.sleep(check_interval)

        self.logger.error(
            f"❌ Server did not recover within {max_wait_seconds}s timeout"
        )
        return False

    def get_crash_history(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve server crash history from system health endpoint.

        Returns:
            Crash history dict or None if unavailable
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/system/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return data.get('crash_history')
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Could not retrieve crash history: {e}")

        return None

    def get_recent_errors(self, lines: int = 50) -> Optional[str]:
        """
        Retrieve recent error log entries.

        Args:
            lines: Number of recent log lines to retrieve

        Returns:
            Error log content or None if unavailable
        """
        try:
            response = requests.get(
                f"{self.api_url}/api/v1/logs/errors.log",
                params={'lines': lines},
                timeout=10
            )
            if response.status_code == 200:
                return response.text
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Could not retrieve error logs: {e}")

        return None

    @property
    def downtime_seconds(self) -> float:
        """Get current downtime in seconds (0 if server is up)."""
        if self._server_down_since is None:
            return 0.0
        return (datetime.now() - self._server_down_since).total_seconds()

    @property
    def consecutive_failures(self) -> int:
        """Get number of consecutive health check failures."""
        return self._consecutive_failures


def with_server_recovery(
    func,
    health_monitor: Rubix44HealthMonitor,
    max_retries: int = 3,
    recovery_timeout: int = 300
):
    """
    Decorator to automatically handle server crashes and recovery.

    Args:
        func: Function to wrap
        health_monitor: Rubix44HealthMonitor instance
        max_retries: Maximum retry attempts
        recovery_timeout: Max seconds to wait for recovery

    Returns:
        Wrapped function with automatic recovery
    """
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                health_monitor.logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}): {e}"
                )

                # Check if server is down
                if not health_monitor.is_healthy():
                    health_monitor.logger.error("Server appears to be down, waiting for recovery...")

                    if health_monitor.wait_for_recovery(max_wait_seconds=recovery_timeout):
                        # Server recovered, retry the operation
                        continue
                    else:
                        # Recovery timeout, give up
                        raise

                # Server is healthy but request failed (transient error)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    health_monitor.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} attempts")

    return wrapper
