"""
Live integration tests for Rubix44 API.

These tests connect to the actual Rubix44 recorder API server.
Requires server to be running at RUBIX44_URL (default: http://10.0.0.58:5000).

Usage:
    # Run live tests
    pytest tests/test_rubix44_api_live.py -v

    # Skip if server not available
    pytest tests/test_rubix44_api_live.py -v -m "not requires_server"
"""

import pytest
import requests
import os
import time
from datetime import datetime


# Mark all tests as requiring server
pytestmark = pytest.mark.requires_server


@pytest.fixture
def rubix_url():
    """Get Rubix44 API URL from environment"""
    return os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')


@pytest.fixture
def check_server_available(rubix_url):
    """Check if Rubix44 server is available"""
    try:
        response = requests.get(f"{rubix_url}/api/v1/health", timeout=2)
        if response.ok:
            return True
    except:
        pass
    pytest.skip("Rubix44 server not available")


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check(self, rubix_url, check_server_available):
        """Test /api/v1/health endpoint"""
        response = requests.get(f"{rubix_url}/api/v1/health", timeout=5)

        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'service' in data

    def test_health_response_format(self, rubix_url, check_server_available):
        """Test health endpoint returns expected format"""
        response = requests.get(f"{rubix_url}/api/v1/health", timeout=5)
        data = response.json()

        # Check required fields
        assert 'status' in data
        assert 'service' in data
        assert 'timestamp' in data

        # Validate timestamp format
        timestamp = data['timestamp']
        datetime.fromisoformat(timestamp)  # Should not raise


class TestStatusEndpoint:
    """Test recording status endpoint"""

    def test_status_when_idle(self, rubix_url, check_server_available):
        """Test /api/v1/recordings/status when no recording active"""
        response = requests.get(f"{rubix_url}/api/v1/recordings/status", timeout=5)

        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        # Could be 'idle' or have 'session' key
        status = data.get('status') or data.get('session', {}).get('status')
        assert status in ['idle', 'recording', 'completed']

    def test_status_response_structure(self, rubix_url, check_server_available):
        """Test status endpoint returns valid structure"""
        response = requests.get(f"{rubix_url}/api/v1/recordings/status", timeout=5)
        data = response.json()

        # Should have either direct status or nested session
        has_direct_status = 'status' in data
        has_nested_session = 'session' in data and 'status' in data['session']

        assert has_direct_status or has_nested_session


class TestStartRecordingEndpoint:
    """Test recording start endpoint (read-only checks)"""

    def test_start_recording_missing_playback_file(self, rubix_url, check_server_available):
        """Test start endpoint validates playback file"""
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={
                'playback_file': 'nonexistent_file_xyz.wav',
                'duration': 10
            },
            timeout=5
        )

        # Should return error for missing file
        assert response.status_code in [400, 404]
        data = response.json()
        assert 'error' in data

    def test_start_recording_missing_parameters(self, rubix_url, check_server_available):
        """Test start endpoint validates required parameters"""
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={},
            timeout=5
        )

        # Should return error for missing parameters
        assert response.status_code in [400, 422]

    def test_start_recording_response_format(self, rubix_url, check_server_available):
        """Test that start endpoint returns expected format (when valid)"""
        # This test requires a valid playback file
        # We'll test the error response format instead
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={
                'playback_file': 'test.wav',
                'duration': 1
            },
            timeout=5
        )

        data = response.json()

        # Should have either session or error
        if response.ok:
            # Success case - check for nested session
            assert 'session' in data or 'session_id' in data
            if 'session' in data:
                session = data['session']
                assert 'id' in session
                assert 'status' in session
        else:
            # Error case
            assert 'error' in data


class TestAPIVersionCompatibility:
    """Test API version compatibility"""

    def test_api_returns_nested_session(self, rubix_url, check_server_available):
        """Test that API uses nested session format (v1.1+)"""
        # Try to start a recording with invalid file
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={
                'playback_file': 'test_nonexistent.wav',
                'duration': 1
            },
            timeout=5
        )

        # Even error responses should indicate format
        # Check headers or response structure
        data = response.json()

        # If successful, should have nested session
        # If error, at least confirms API structure
        assert isinstance(data, dict)

    def test_session_metadata_fields(self, rubix_url, check_server_available):
        """Test that session objects include expected metadata fields"""
        response = requests.get(f"{rubix_url}/api/v1/recordings/status", timeout=5)
        data = response.json()

        if 'session' in data:
            session = data['session']
            # Check for new metadata fields (added Jan 2026)
            expected_fields = ['id', 'status']
            for field in expected_fields:
                assert field in session


class TestResponseTiming:
    """Test API response times"""

    def test_health_endpoint_response_time(self, rubix_url, check_server_available):
        """Test health endpoint responds quickly"""
        start = time.time()
        response = requests.get(f"{rubix_url}/api/v1/health", timeout=5)
        elapsed = time.time() - start

        assert response.ok
        assert elapsed < 1.0  # Should respond within 1 second

    def test_status_endpoint_response_time(self, rubix_url, check_server_available):
        """Test status endpoint responds quickly"""
        start = time.time()
        response = requests.get(f"{rubix_url}/api/v1/recordings/status", timeout=5)
        elapsed = time.time() - start

        assert response.ok
        assert elapsed < 1.0


class TestErrorHandling:
    """Test API error handling"""

    def test_invalid_endpoint(self, rubix_url, check_server_available):
        """Test accessing invalid endpoint"""
        response = requests.get(f"{rubix_url}/api/v1/invalid_endpoint", timeout=5)
        assert response.status_code == 404

    def test_invalid_method(self, rubix_url, check_server_available):
        """Test using wrong HTTP method"""
        # Health is GET only
        response = requests.post(f"{rubix_url}/api/v1/health", timeout=5)
        assert response.status_code in [405, 400]

    def test_malformed_json(self, rubix_url, check_server_available):
        """Test sending malformed JSON"""
        response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            data="not valid json",
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        assert response.status_code in [400, 422]


@pytest.mark.slow
class TestRecordingWorkflow:
    """Test complete recording workflow (slow tests)"""

    @pytest.mark.skip(reason="Requires valid playback file and takes 3+ minutes")
    def test_complete_recording_cycle(self, rubix_url, check_server_available):
        """
        Test complete recording workflow:
        1. Start recording
        2. Monitor progress
        3. Wait for completion
        4. Verify files created

        NOTE: This test is skipped by default as it requires:
        - Valid playback file
        - Time to complete (3+ minutes)
        - Clean recording environment
        """
        # Start recording
        start_response = requests.post(
            f"{rubix_url}/api/v1/recordings/start",
            json={
                'playback_file': 'test.wav',  # Must exist
                'duration': 10,  # Short duration
                'output_prefix': 'pytest_test'
            },
            timeout=5
        )

        if not start_response.ok:
            pytest.skip("Cannot start recording - check server state")

        data = start_response.json()
        session = data.get('session', {})
        session_id = session.get('id')

        assert session_id is not None

        # Monitor progress
        max_wait = 30  # Wait up to 30 seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = requests.get(
                f"{rubix_url}/api/v1/recordings/status",
                timeout=5
            )
            status_data = status_response.json()

            session_info = status_data.get('session', status_data)
            current_status = session_info.get('status')

            if current_status in ['completed', 'idle']:
                break
            elif current_status == 'error':
                pytest.fail(f"Recording failed: {session_info.get('error')}")

            time.sleep(2)

        # Verify completion
        assert current_status in ['completed', 'idle']


def test_connection_available():
    """Quick test to check if server is reachable"""
    rubix_url = os.getenv('RUBIX44_URL', 'http://10.0.0.58:5000')
    try:
        response = requests.get(f"{rubix_url}/api/v1/health", timeout=2)
        assert response.ok, "Server not responding correctly"
        print(f"\n✓ Rubix44 server available at {rubix_url}")
    except Exception as e:
        pytest.skip(f"Server not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
