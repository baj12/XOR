"""
Test suite for advanced visualization functions.

Tests all new visualization capabilities:
- Embedding comparison plots (before/after network)
- Layer-by-layer progression
- Classification curves (ROC, PR, threshold analysis)
- Audio feature analysis
- Model comparison dashboard
- Weight evolution plots
- Extended universal plots (PCA, t-SNE)
- Statistical analysis plots
- Hyperparameter impact visualization
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


class TestAdvancedVisualizations(unittest.TestCase):
    """Test suite for advanced_visualizations.py functions"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used by all tests"""
        # Create temporary directory for outputs
        cls.temp_dir = tempfile.mkdtemp()

        # Generate synthetic XOR-like data
        np.random.seed(42)
        cls.n_samples = 500
        cls.n_features = 10

        # Create binary classification dataset
        X, y = make_classification(
            n_samples=cls.n_samples,
            n_features=cls.n_features,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )

        # Split data
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create a simple test model
        cls.model = keras.Sequential([
            keras.layers.Input(shape=(cls.n_features,)),
            keras.layers.Dense(8, activation='relu', name='hidden_1'),
            keras.layers.Dense(4, activation='relu', name='hidden_2'),
            keras.layers.Dense(1, activation='sigmoid', name='output')
        ])

        cls.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train briefly
        cls.model.fit(
            cls.X_train, cls.y_train,
            validation_data=(cls.X_test, cls.y_test),
            epochs=5,
            batch_size=32,
            verbose=0
        )

        # Generate predictions
        cls.y_pred_proba = cls.model.predict(cls.X_test, verbose=0)
        cls.y_pred = (cls.y_pred_proba > 0.5).astype(int).flatten()
        cls.y_pred_proba = cls.y_pred_proba.flatten()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        shutil.rmtree(cls.temp_dir)

    def test_embedding_comparison_plots_xor_mode(self):
        """Test create_embedding_comparison_plots() for XOR mode"""
        from embedding_analysis_plots import create_embedding_comparison_plots

        output_path = os.path.join(self.temp_dir, 'embedding_comparison_xor.png')

        # Run function
        create_embedding_comparison_plots(
            self.model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            output_path,
            mode='xor'
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Embedding comparison plot not created")

        # Verify file size (should be non-trivial)
        file_size = os.path.getsize(output_path)
        self.assertGreater(file_size, 10000, "Plot file seems too small")

    def test_embedding_comparison_plots_audio_mode(self):
        """Test create_embedding_comparison_plots() for Audio mode"""
        from embedding_analysis_plots import create_embedding_comparison_plots

        output_path = os.path.join(self.temp_dir, 'embedding_comparison_audio.png')

        # Run function
        create_embedding_comparison_plots(
            self.model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            output_path,
            mode='audio'
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Audio embedding comparison plot not created")

    def test_layer_progression_plots(self):
        """Test create_layer_progression_plots()"""
        from embedding_analysis_plots import create_layer_progression_plots

        output_path = os.path.join(self.temp_dir, 'layer_progression.png')

        # Use smaller sample for speed
        X_sample = self.X_test[:200]
        y_sample = self.y_test[:200]

        # Run function
        create_layer_progression_plots(
            self.model,
            X_sample,
            y_sample,
            output_path,
            max_samples=200
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Layer progression plot not created")

        # Verify file size
        file_size = os.path.getsize(output_path)
        self.assertGreater(file_size, 10000, "Plot file seems too small")

    def test_classification_curves(self):
        """Test create_classification_curves()"""
        from embedding_analysis_plots import create_classification_curves

        output_path = os.path.join(self.temp_dir, 'classification_curves.png')

        # Run function
        create_classification_curves(
            self.y_test,
            self.y_pred_proba,
            output_path,
            class_names=['Class 0', 'Class 1']
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Classification curves plot not created")

    def test_audio_feature_analysis(self):
        """Test create_audio_feature_analysis()"""
        from embedding_analysis_plots import create_audio_feature_analysis

        output_path = os.path.join(self.temp_dir, 'audio_feature_analysis.png')

        # Create mock audio feature data
        # Simulate MFCC features: (n_samples, n_mfcc * n_frames)
        n_mfcc = 13
        n_frames = 40
        n_samples = 200

        # Generate synthetic MFCC-like features
        features = np.random.randn(n_samples, n_mfcc * n_frames)
        labels = np.random.randint(0, 2, n_samples)

        # Feature names
        feature_names = [f'mfcc_{i}' for i in range(n_mfcc)]

        # Run function
        create_audio_feature_analysis(
            features,
            feature_names,
            labels,
            output_path
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Audio feature analysis plot not created")

    def test_model_comparison_dashboard(self):
        """Test create_model_comparison_dashboard()"""
        from embedding_analysis_plots import create_model_comparison_dashboard

        output_path = os.path.join(self.temp_dir, 'model_comparison.png')

        # Create a second model for comparison
        model2 = keras.Sequential([
            keras.layers.Input(shape=(self.n_features,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model2.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model2.fit(
            self.X_train, self.y_train,
            epochs=3,
            batch_size=32,
            verbose=0
        )

        models_dict = {
            'Model 1 (3 layers)': self.model,
            'Model 2 (4 layers)': model2
        }

        # Run function
        create_model_comparison_dashboard(
            models_dict,
            self.X_test,
            self.y_test,
            output_path
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Model comparison dashboard not created")

    def test_weight_evolution_plots(self):
        """Test create_weight_evolution_plots()"""
        from embedding_analysis_plots import create_weight_evolution_plots

        output_path = os.path.join(self.temp_dir, 'weight_evolution.png')

        # Create mock weight history (3 checkpoints)
        history_checkpoints = []

        for epoch in [0, 5, 10]:
            # Get current weights
            weights = self.model.get_weights()
            history_checkpoints.append({
                'epoch': epoch,
                'weights': [w.copy() for w in weights]
            })

        # Run function
        create_weight_evolution_plots(
            history_checkpoints,
            output_path
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Weight evolution plot not created")


class TestUniversalPlotsExtensions(unittest.TestCase):
    """Test suite for extensions to universal_plots.py"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()

        # Generate synthetic data
        np.random.seed(42)
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42
        )

        cls.X_train, cls.X_val, cls.y_train, cls.y_val = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Create and train model
        cls.model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        cls.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        cls.model.fit(
            cls.X_train, cls.y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )

        cls.y_pred_proba = cls.model.predict(cls.X_val, verbose=0).flatten()
        cls.y_pred = (cls.y_pred_proba > 0.5).astype(int)

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_universal_plots_with_pca(self):
        """Test create_universal_classification_plots() with PCA enabled"""
        from universal_plots import create_universal_classification_plots

        output_path = os.path.join(self.temp_dir, 'universal_with_pca.png')

        # Run function with PCA
        create_universal_classification_plots(
            self.model,
            self.X_train,
            self.X_val,
            self.y_train,
            self.y_val,
            output_path,
            title_prefix="Test with PCA",
            include_pca=True,
            include_tsne=False
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Universal plots with PCA not created")

    def test_universal_plots_with_tsne(self):
        """Test create_universal_classification_plots() with t-SNE enabled"""
        from universal_plots import create_universal_classification_plots

        output_path = os.path.join(self.temp_dir, 'universal_with_tsne.png')

        # Run function with t-SNE (using small sample for speed)
        create_universal_classification_plots(
            self.model,
            self.X_train[:200],
            self.X_val[:100],
            self.y_train[:200],
            self.y_val[:100],
            output_path,
            title_prefix="Test with t-SNE",
            include_pca=False,
            include_tsne=True
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Universal plots with t-SNE not created")

    def test_statistical_analysis_plots(self):
        """Test create_statistical_analysis_plots()"""
        from universal_plots import create_statistical_analysis_plots

        output_path = os.path.join(self.temp_dir, 'statistical_analysis.png')

        # Run function
        create_statistical_analysis_plots(
            self.y_val,
            self.y_pred,
            self.y_pred_proba,
            output_path
        )

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Statistical analysis plots not created")


class TestGAVisualizationExtensions(unittest.TestCase):
    """Test suite for GA visualization extensions"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_hyperparameter_impact_plot(self):
        """Test create_hyperparameter_impact_plot()"""
        from genetic_algorithm import GAProgressPlotter
        from utils import GAConfig, ModelConfig

        output_path = os.path.join(self.temp_dir, 'hyperparameter_impact.png')

        # Create mock logbook (GA history)
        logbook = []
        for gen in range(20):
            logbook.append({
                'gen': gen,
                'avg': 0.5 + 0.02 * gen + np.random.randn() * 0.01,
                'max': 0.6 + 0.015 * gen + np.random.randn() * 0.01,
                'min': 0.4 + 0.01 * gen + np.random.randn() * 0.01,
                'std': 0.1 - 0.003 * gen + np.random.randn() * 0.005
            })

        # Create mock config
        class MockConfig:
            def __init__(self):
                self.ga = GAConfig(
                    population_size=20,
                    ngen=20,
                    cxpb=0.7,
                    mutpb=0.2,
                    epochs=10,
                    n_processes=1,
                    max_time_per_ind=300.0
                )
                self.model = ModelConfig(
                    hidden_layers=[8, 4],
                    neurons_per_layer=8,
                    lr=0.001,
                    activation='relu',
                    optimizer='adam',
                    batch_size=32,
                    skip_connections='none',
                    input_dim=10
                )

        config = MockConfig()

        # Create plotter with required arguments
        class MockPaths:
            def __init__(self, temp_dir):
                self.plots = temp_dir
                self.temp_dir = temp_dir

        mock_paths = MockPaths(self.temp_dir)

        plotter = GAProgressPlotter(
            plot_path=os.path.join(self.temp_dir, 'ga_progress.png'),
            paths=mock_paths
        )
        plotter.create_hyperparameter_impact_plot(logbook, config, output_path)

        # Verify output exists
        self.assertTrue(os.path.exists(output_path), "Hyperparameter impact plot not created")


class TestIntegration(unittest.TestCase):
    """Integration tests for visualization pipeline"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_full_visualization_pipeline(self):
        """Test that all visualizations work together"""
        from embedding_analysis_plots import (
            create_embedding_comparison_plots,
            create_layer_progression_plots,
            create_classification_curves
        )
        from universal_plots import create_statistical_analysis_plots

        # Generate data
        np.random.seed(42)
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create model
        model = keras.Sequential([
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(4, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=5,
            batch_size=32,
            verbose=0
        )

        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Run all visualizations
        viz_configs = [
            ('embedding_comparison.png', lambda p: create_embedding_comparison_plots(
                model, X_train, X_test, y_train, y_test, p, mode='xor'
            )),
            ('layer_progression.png', lambda p: create_layer_progression_plots(
                model, X_test[:200], y_test[:200], p, max_samples=200
            )),
            ('classification_curves.png', lambda p: create_classification_curves(
                y_test, y_pred_proba, p
            )),
            ('statistical_analysis.png', lambda p: create_statistical_analysis_plots(
                y_test, y_pred, y_pred_proba, p
            ))
        ]

        for filename, viz_func in viz_configs:
            output_path = os.path.join(self.temp_dir, filename)
            try:
                viz_func(output_path)
                self.assertTrue(
                    os.path.exists(output_path),
                    f"Integration test failed: {filename} not created"
                )
            except Exception as e:
                self.fail(f"Integration test failed for {filename}: {e}")


def run_tests(verbosity=2):
    """Run all tests with specified verbosity"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedVisualizations))
    suite.addTests(loader.loadTestsFromTestCase(TestUniversalPlotsExtensions))
    suite.addTests(loader.loadTestsFromTestCase(TestGAVisualizationExtensions))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    # Run tests when script is executed directly
    result = run_tests(verbosity=2)

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
