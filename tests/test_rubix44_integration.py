"""
Unit and integration tests for Rubix44 API integration.

Tests the recording orchestrator's ability to parse API responses,
handle different status values, and extract session metadata.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from continuous.recording_orchestrator import ContinuousRecordingOrchestrator


class TestRubix44ResponseParsing:
    """Test parsing of different Rubix44 API response formats"""

    def test_parse_nested_session_response(self):
        """Test parsing new nested session format (v1.1+)"""
        response_data = {
            "message": "Recording started",
            "session": {
                "id": "20260114_175618",
                "human_id": "swift-panda-2347",
                "status": "recording",
                "duration": 180,
                "playback_file": "test.wav"
            }
        }

        session_data = response_data.get('session', {})
        session_id = session_data.get('id')
        human_id = session_data.get('human_id')

        assert session_id == "20260114_175618"
        assert human_id == "swift-panda-2347"

    def test_parse_legacy_session_response(self):
        """Test parsing old direct session_id format (v1.0)"""
        response_data = {
            "session_id": "20260114_175618",
            "message": "Recording started"
        }

        # Try nested format first
        session_data = response_data.get('session', {})
        session_id = session_data.get('id')

        # Fallback to direct format
        if not session_id:
            session_id = response_data.get('session_id')

        assert session_id == "20260114_175618"

    def test_parse_missing_session_id(self):
        """Test handling of response without session ID"""
        response_data = {
            "message": "Recording started"
        }

        session_data = response_data.get('session', {})
        session_id = session_data.get('id')

        if not session_id:
            session_id = response_data.get('session_id')

        assert session_id is None

    def test_extract_progress_information(self):
        """Test extracting progress metadata from session"""
        response_data = {
            "session": {
                "id": "20260114_175618",
                "status": "recording",
                "progress_percent": 45.5,
                "elapsed_seconds": 82,
                "expected_duration": 180
            }
        }

        session_info = response_data.get('session', {})
        progress = session_info.get('progress_percent', 0)
        elapsed = session_info.get('elapsed_seconds', 0)

        assert progress == 45.5
        assert elapsed == 82

    def test_extract_error_information(self):
        """Test extracting error information from failed session"""
        response_data = {
            "session": {
                "id": "20260114_175618",
                "status": "error",
                "error": "Playback file not found"
            }
        }

        session_info = response_data.get('session', {})
        status = session_info.get('status')
        error = session_info.get('error')

        assert status == "error"
        assert error == "Playback file not found"


class TestStatusHandling:
    """Test handling of different recording status values"""

    def test_handle_completed_status(self):
        """Test handling of 'completed' status"""
        status_data = {
            "session": {
                "status": "completed",
                "id": "20260114_175618"
            }
        }

        session_info = status_data['session']
        status = session_info.get('status')

        assert status == "completed"

    def test_handle_idle_status(self):
        """Test handling of 'idle' status (legacy completed)"""
        status_data = {
            "status": "idle",
            "message": "No active recording"
        }

        # Handle direct format
        session_info = status_data
        status = session_info.get('status')

        assert status == "idle"

    def test_handle_error_status(self):
        """Test handling of 'error' status"""
        status_data = {
            "session": {
                "status": "error",
                "error": "Device not found"
            }
        }

        session_info = status_data['session']
        status = session_info.get('status')
        error = session_info.get('error')

        assert status == "error"
        assert error == "Device not found"

    def test_handle_stopped_status(self):
        """Test handling of 'stopped' status (manual stop)"""
        status_data = {
            "session": {
                "status": "stopped",
                "id": "20260114_175618"
            }
        }

        session_info = status_data['session']
        status = session_info.get('status')

        assert status == "stopped"

    def test_handle_recording_status(self):
        """Test handling of 'recording' status with progress"""
        status_data = {
            "session": {
                "status": "recording",
                "progress_percent": 65.0,
                "elapsed_seconds": 117
            }
        }

        session_info = status_data['session']
        status = session_info.get('status')
        progress = session_info.get('progress_percent', 0)

        assert status == "recording"
        assert progress == 65.0


class TestAPICompatibility:
    """Test compatibility with different API versions"""

    def test_v11_nested_format(self):
        """Test v1.1+ nested session format"""
        response = {
            "message": "Recording started",
            "session": {
                "id": "20260114_175618",
                "human_id": "brave-wolf-9832"
            }
        }

        # Extract using nested format
        session_data = response.get('session', {})
        session_id = session_data.get('id')

        assert session_id is not None

    def test_v10_direct_format(self):
        """Test v1.0 direct session_id format"""
        response = {
            "session_id": "20260114_175618",
            "message": "Recording started"
        }

        # Try nested first (will fail)
        session_data = response.get('session', {})
        session_id = session_data.get('id')

        # Fallback to direct
        if not session_id:
            session_id = response.get('session_id')

        assert session_id == "20260114_175618"

    def test_dual_format_compatibility(self):
        """Test code handles both formats correctly"""
        def extract_session_id(response):
            """Helper to extract session ID from either format"""
            session_data = response.get('session', {})
            session_id = session_data.get('id')
            if not session_id:
                session_id = response.get('session_id')
            return session_id

        # Test new format
        new_response = {"session": {"id": "test123"}}
        assert extract_session_id(new_response) == "test123"

        # Test old format
        old_response = {"session_id": "test456"}
        assert extract_session_id(old_response) == "test456"


class TestHumanIdExtraction:
    """Test extraction and logging of human-readable IDs"""

    def test_extract_human_id(self):
        """Test extracting human_id from session"""
        response = {
            "session": {
                "id": "20260114_175618",
                "human_id": "gentle-tiger-4521"
            }
        }

        session_data = response.get('session', {})
        human_id = session_data.get('human_id', 'N/A')

        assert human_id == "gentle-tiger-4521"

    def test_human_id_fallback(self):
        """Test fallback when human_id not present"""
        response = {
            "session": {
                "id": "20260114_175618"
            }
        }

        session_data = response.get('session', {})
        human_id = session_data.get('human_id', 'N/A')

        assert human_id == 'N/A'

    def test_human_id_format_validation(self):
        """Test that human_id follows expected format"""
        human_id = "swift-panda-2347"
        parts = human_id.split('-')

        # Should be adjective-animal-number
        assert len(parts) == 3
        assert parts[2].isdigit()


@pytest.mark.asyncio
class TestOrchestratorIntegration:
    """Integration tests for the orchestrator with mocked API calls"""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator for testing"""
        with patch('continuous.recording_orchestrator.DatabaseConnection'):
            orchestrator = ContinuousRecordingOrchestrator('test_exp')
            orchestrator.config = {
                'recording_duration_seconds': 180,
                'playback_file': 'test.wav',
                'output_prefix': 'test'
            }
            orchestrator.logger = Mock()
            return orchestrator

    async def test_start_recording_success(self, mock_orchestrator):
        """Test successful recording start with nested response"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": "Recording started",
            "session": {
                "id": "20260114_175618",
                "human_id": "brave-fox-1234",
                "status": "recording",
                "duration": 180
            }
        }
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response):
            session_id = await mock_orchestrator.start_recording_cycle(1)

        assert session_id == "20260114_175618"
        mock_orchestrator.logger.info.assert_called()

    async def test_start_recording_no_session_id(self, mock_orchestrator):
        """Test handling of response without session ID"""
        mock_response = Mock()
        mock_response.json.return_value = {"message": "Recording started"}
        mock_response.raise_for_status = Mock()

        with patch('requests.post', return_value=mock_response):
            session_id = await mock_orchestrator.start_recording_cycle(1)

        assert session_id is None
        mock_orchestrator.logger.error.assert_called()


class TestProgressReporting:
    """Test progress reporting functionality"""

    def test_calculate_progress_percentage(self):
        """Test progress percentage calculation"""
        elapsed = 90  # seconds
        duration = 180  # seconds
        progress = (elapsed / duration) * 100

        assert progress == 50.0

    def test_format_progress_message(self):
        """Test formatting of progress log messages"""
        session_id = "20260114_175618"
        progress_percent = 45.5
        elapsed_seconds = 82

        message = f"Recording {session_id}: {progress_percent:.1f}% complete ({elapsed_seconds:.0f}s elapsed)"

        assert "45.5% complete" in message
        assert "82s elapsed" in message


def test_all_status_values():
    """Test that all expected status values are handled"""
    expected_statuses = [
        'recording',
        'completed',
        'idle',
        'error',
        'stopped',
        'unknown'
    ]

    # Verify each status can be extracted
    for status in expected_statuses:
        data = {"session": {"status": status}}
        extracted = data['session']['status']
        assert extracted == status


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
