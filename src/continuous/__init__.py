"""
Continuous Learning Module

Handles continuous audio classification from stereo WAV streams where:
- Left channel = Positive class samples
- Right channel = Negative class samples
"""

from .stereo_channel_processor import StereoChannelProcessor, simulate_continuous_stream
from .feature_database import FeatureDatabase
from .continuous_ingestion import ContinuousIngestionPipeline, SimulatedContinuousStream
from .incremental_trainer import IncrementalTrainer
from .monitoring_reporter import SystemMonitor, WeeklyReport
from .orchestrator import ContinuousLearningOrchestrator, EmailReporter
from .visualization import ContinuousVisualizer
from .model_manager import ModelManager

__all__ = [
    'StereoChannelProcessor',
    'FeatureDatabase',
    'simulate_continuous_stream',
    'ContinuousIngestionPipeline',
    'SimulatedContinuousStream',
    'IncrementalTrainer',
    'SystemMonitor',
    'WeeklyReport',
    'ContinuousLearningOrchestrator',
    'EmailReporter',
    'ContinuousVisualizer',
    'ModelManager'
]
