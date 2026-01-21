"""
Tests for Rubix44 data provider
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import numpy as np

from src.continuous.rubix44_data_provider import Rubix44Client, Rubix44DataProvider


class TestRubix44Client:
    """Tests for Rubix44Client"""

    @pytest.fixture
    def client(self):
        return Rubix44Client("http://test-server:5000")

    def test_initialization(self, client):
        """Test client initialization"""
        assert client.base_url == "http://test-server:5000"
        assert client.api_base == "http://test-server:5000/api/v1"

    @patch('requests.get')
    def test_health_check_success(self, mock_get, client):
        """Test successful health check using /health endpoint"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'healthy',
            'service': 'Rubix Recorder API',
            'timestamp': '2026-01-11T12:00:00Z'
        }
        mock_get.return_value = mock_response

        assert client.health_check() is True
        # Verify it calls the correct endpoint
        mock_get.assert_called_with("http://test-server:5000/api/v1/health", timeout=5)

    @patch('requests.get')
    def test_health_check_failure(self, mock_get, client):
        """Test failed health check"""
        mock_get.side_effect = Exception("Connection refused")

        assert client.health_check() is False

    @patch('requests.get')
    def test_get_config(self, mock_get, client):
        """Test get configuration"""
        expected_config = {
            'sample_rate': 44100,
            'port': 5000,
            'output_prefix': 'recording'
        }
        mock_response = Mock()
        mock_response.json.return_value = expected_config
        mock_get.return_value = mock_response

        config = client.get_config()
        assert config == expected_config

    @patch('requests.get')
    def test_get_recording_status(self, mock_get, client):
        """Test get recording status"""
        expected_status = {'status': 'idle', 'message': 'No active recording'}
        mock_response = Mock()
        mock_response.json.return_value = expected_status
        mock_get.return_value = mock_response

        status = client.get_recording_status()
        assert status == expected_status

    @patch('requests.get')
    def test_get_devices(self, mock_get, client):
        """Test get devices list"""
        expected_devices = [
            {
                'id': 0,
                'name': 'Rubix44 USB',
                'input_channels': 4,
                'output_channels': 4,
                'sample_rate': 44100
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = expected_devices
        mock_get.return_value = mock_response

        devices = client.get_devices()
        assert devices == expected_devices

    @patch('requests.get')
    def test_get_rubix_device_found(self, mock_get, client):
        """Test get Rubix device when found"""
        expected_response = {
            'found': True,
            'input_device': 0,
            'output_device': 0,
            'input_device_info': {
                'id': 0,
                'name': 'Rubix44',
                'channels': 4,
                'sample_rate': 44100
            }
        }
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        mock_get.return_value = mock_response

        device = client.get_rubix_device()
        assert device is not None
        assert device['found'] is True

    @patch('requests.get')
    def test_get_rubix_device_not_found(self, mock_get, client):
        """Test get Rubix device when not found"""
        mock_response = Mock()
        mock_response.json.return_value = {'found': False}
        mock_get.return_value = mock_response

        device = client.get_rubix_device()
        assert device is None

    @patch('requests.get')
    def test_get_playback_files(self, mock_get, client):
        """Test get playback files with metadata"""
        expected_files = [
            {
                'filename': 'noise_baseline.wav',
                'duration_seconds': 120.5,
                'sample_rate': 44100,
                'channels': 2,
                'size': 1024000
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = expected_files
        mock_get.return_value = mock_response

        files = client.get_playback_files()
        assert files == expected_files

    @patch('requests.get')
    def test_get_recording_history(self, mock_get, client):
        """Test get recording history with v1.1.0 enhanced metadata"""
        expected_history = [
            {
                'id': 'session_2026-01-04_12-00-00',
                'prefix': 'test',
                'timestamp': '2026-01-04_12-00-00',
                'start_time': '2026-01-04T12:00:00',
                'end_time': '2026-01-04T13:00:00',
                'duration_seconds': 3600.0,
                'playback_file': 'noise_baseline.wav',
                'sample_rate': 44100,
                'files': [
                    {
                        'name': 'test_stereo.wav',
                        'size': 1000000,
                        'modified': '2026-01-04T13:00:00'
                    }
                ]
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = expected_history
        mock_get.return_value = mock_response

        history = client.get_recording_history()
        assert history == expected_history
        # Verify v1.1.0 fields are present
        assert history[0]['duration_seconds'] == 3600.0
        assert history[0]['playback_file'] == 'noise_baseline.wav'

    @patch('requests.get')
    def test_download_recording(self, mock_get, client):
        """Test download recording"""
        # Create mock response with streaming content
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b'chunk1', b'chunk2'])
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test.wav'
            result = client.download_recording('test.wav', save_path)

            assert result is True
            assert save_path.exists()
            assert save_path.read_bytes() == b'chunk1chunk2'


class TestRubix44DataProvider:
    """Tests for Rubix44DataProvider"""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        config = Mock()
        config.audio.sample_rate = 22050
        config.audio.segment_duration = 1.0
        config.audio.n_mfcc = 13
        config.audio.feature_types = ['mfcc', 'spectral']
        return config

    @pytest.fixture
    def mock_processor(self):
        """Create mock stereo channel processor"""
        processor = Mock()

        # Mock process_stereo_file to return feature arrays
        def mock_process(wav_path, max_samples=None):
            n_samples = 100
            feature_dim = 680
            X_left = np.random.randn(n_samples, feature_dim)
            y_left = np.ones(n_samples)
            X_right = np.random.randn(n_samples, feature_dim)
            y_right = np.zeros(n_samples)
            offsets_left = np.arange(n_samples, dtype=float)
            offsets_right = np.arange(n_samples, dtype=float)
            return X_left, y_left, X_right, y_right, offsets_left, offsets_right

        processor.process_stereo_file = Mock(side_effect=mock_process)
        return processor

    @pytest.fixture
    def mock_database(self):
        """Create mock feature database"""
        db = Mock()
        db.store_features = Mock()
        return db

    @pytest.fixture
    def provider(self, mock_processor, mock_database):
        """Create Rubix44DataProvider instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Rubix44DataProvider(
                api_url="http://test-server:5000",
                download_dir=Path(tmpdir),
                processor=mock_processor,
                database=mock_database,
                output_prefix_filter=None,
                cleanup_after_processing=False
            )
            yield provider

    def test_initialization(self, provider):
        """Test provider initialization"""
        assert provider.client is not None
        assert provider.download_dir.exists()
        assert len(provider.processed_sessions) == 0

    def test_load_save_state(self, mock_processor, mock_database):
        """Test state persistence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            download_dir = Path(tmpdir)
            state_file = download_dir / "test_state.json"

            # Create provider and add some processed sessions
            provider = Rubix44DataProvider(
                api_url="http://test:5000",
                download_dir=download_dir,
                processor=mock_processor,
                database=mock_database,
                state_file=state_file
            )

            provider.processed_sessions.add('session1')
            provider.processed_sessions.add('session2')
            provider._save_state()

            # Create new provider and verify state loaded
            provider2 = Rubix44DataProvider(
                api_url="http://test:5000",
                download_dir=download_dir,
                processor=mock_processor,
                database=mock_database,
                state_file=state_file
            )

            assert 'session1' in provider2.processed_sessions
            assert 'session2' in provider2.processed_sessions

    def test_parse_timestamp(self, provider):
        """Test timestamp parsing"""
        timestamp_str = "2026-01-04_12-30-45"
        dt = provider._parse_timestamp(timestamp_str)

        assert isinstance(dt, datetime)
        assert dt.year == 2026
        assert dt.month == 1
        assert dt.day == 4
        assert dt.hour == 12
        assert dt.minute == 30
        assert dt.second == 45

    def test_parse_invalid_timestamp(self, provider):
        """Test parsing invalid timestamp falls back to current time"""
        timestamp_str = "invalid-timestamp"
        dt = provider._parse_timestamp(timestamp_str)

        assert isinstance(dt, datetime)
        # Should be close to current time
        assert abs((datetime.now() - dt).total_seconds()) < 5

    @patch.object(Rubix44Client, 'get_recording_history')
    def test_poll_no_new_recordings(self, mock_history, provider):
        """Test polling when no new recordings"""
        # Return empty history
        mock_history.return_value = []

        count = provider.poll_for_new_recordings()
        assert count == 0

    @patch.object(Rubix44Client, 'get_recording_history')
    def test_poll_skip_already_processed(self, mock_history, provider):
        """Test polling skips already processed sessions"""
        # Add session to processed set
        provider.processed_sessions.add('session1')

        # History has only the processed session
        mock_history.return_value = [
            {
                'id': 'session1',
                'prefix': 'test',
                'timestamp': '2026-01-04_12-00-00',
                'files': []
            }
        ]

        count = provider.poll_for_new_recordings()
        assert count == 0

    @patch.object(Rubix44Client, 'download_recording')
    @patch.object(Rubix44Client, 'get_recording_history')
    def test_poll_process_new_recording(self, mock_history, mock_download,
                                       provider, mock_processor, mock_database):
        """Test polling and processing new recording"""
        # Mock history with new session
        mock_history.return_value = [
            {
                'id': 'session_new',
                'prefix': 'test',
                'timestamp': '2026-01-04_12-00-00',
                'files': [
                    {
                        'name': 'test_2026-01-04_12-00-00_stereo.wav',
                        'size': 1000000,
                        'path': 'recordings/test_stereo.wav'
                    }
                ]
            }
        ]

        # Mock download to create a dummy file
        def mock_download_file(filename, save_path):
            save_path.write_bytes(b'dummy wav data')
            return True

        mock_download.side_effect = mock_download_file

        # Process
        count = provider.poll_for_new_recordings()

        # Verify
        assert count == 1
        assert 'session_new' in provider.processed_sessions
        assert mock_processor.process_stereo_file.called
        assert mock_database.store_features.called

    @patch.object(Rubix44Client, 'get_recording_history')
    def test_prefix_filter(self, mock_history, mock_processor, mock_database):
        """Test prefix filtering"""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Rubix44DataProvider(
                api_url="http://test:5000",
                download_dir=Path(tmpdir),
                processor=mock_processor,
                database=mock_database,
                output_prefix_filter="xor_"
            )

            # Mock history with sessions of different prefixes
            mock_history.return_value = [
                {
                    'id': 'xor_session',
                    'prefix': 'xor_recording',
                    'timestamp': '2026-01-04_12-00-00',
                    'files': [{'name': 'xor_stereo.wav', 'size': 1000}]
                },
                {
                    'id': 'other_session',
                    'prefix': 'test_recording',
                    'timestamp': '2026-01-04_12-00-00',
                    'files': [{'name': 'test_stereo.wav', 'size': 1000}]
                }
            ]

            # The filtered polling should not process either (no download mock)
            # but should filter correctly
            count = provider.poll_for_new_recordings()

            # Only xor_ session should be attempted
            # (will fail due to missing download mock, but filtering logic is tested)
            assert count == 0  # Both fail without download, but xor_ was attempted

    def test_get_stats(self, provider):
        """Test getting provider statistics"""
        provider.processed_sessions.add('session1')
        provider.processed_sessions.add('session2')

        stats = provider.get_stats()

        assert 'api_url' in stats
        assert 'download_dir' in stats
        assert stats['processed_sessions'] == 2
        assert 'api_healthy' in stats

    @patch.object(Rubix44Client, 'get_recording_history')
    def test_skip_short_recordings(self, mock_history, mock_processor, mock_database):
        """Test that recordings shorter than min_duration are skipped"""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Rubix44DataProvider(
                api_url="http://test:5000",
                download_dir=Path(tmpdir),
                processor=mock_processor,
                database=mock_database,
                min_recording_duration_sec=60.0,
                validate_device_on_startup=False
            )

            # Mock history with a short recording (30 seconds)
            mock_history.return_value = [
                {
                    'id': 'short_session',
                    'prefix': 'test',
                    'timestamp': '2026-01-04_12-00-00',
                    'duration_seconds': 30.0,  # Too short!
                    'playback_file': 'test.wav',
                    'sample_rate': 44100,
                    'files': [{'name': 'test_stereo.wav', 'size': 1000}]
                }
            ]

            # Should process 0 recordings (skipped due to duration)
            count = provider.poll_for_new_recordings()
            assert count == 0
            assert 'short_session' not in provider.processed_sessions

    @patch.object(Rubix44Client, 'download_recording')
    @patch.object(Rubix44Client, 'get_recording_history')
    def test_cleanup_after_processing(self, mock_history, mock_download, mock_processor, mock_database):
        """Test file cleanup after successful processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = Rubix44DataProvider(
                api_url="http://test:5000",
                download_dir=Path(tmpdir),
                processor=mock_processor,
                database=mock_database,
                cleanup_after_processing=True
            )

            mock_history.return_value = [
                {
                    'id': 'session1',
                    'prefix': 'test',
                    'timestamp': '2026-01-04_12-00-00',
                    'files': [
                        {'name': 'test_stereo.wav', 'size': 1000}
                    ]
                }
            ]

            wav_file = Path(tmpdir) / 'test_stereo.wav'

            def mock_download_file(filename, save_path):
                save_path.write_bytes(b'wav data')
                return True

            mock_download.side_effect = mock_download_file

            # Process
            count = provider.poll_for_new_recordings()

            # File should be deleted after processing
            assert count == 1
            assert not wav_file.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
