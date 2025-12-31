"""
Continuous Learning Module

Handles continuous audio classification from stereo WAV streams where:
- Left channel = Positive class samples
- Right channel = Negative class samples
"""

from .stereo_channel_processor import StereoChannelProcessor, simulate_continuous_stream
from .feature_database import FeatureDatabase

__all__ = [
    'StereoChannelProcessor',
    'FeatureDatabase',
    'simulate_continuous_stream'
]
