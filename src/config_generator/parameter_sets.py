# parameter_sets.py

# Architecture parameters
NOISE_DIMENSIONS = [0, 1, 2, 5, 10, 20]  # Different levels of noise complexity
CLASS_SEPARATION = {
    'clear': 0.5,      # Clear separation
    'partial': 0.75,   # Some overlap
    'complete': 1.0    # Maximum separation
}
HIDDEN_LAYERS_RANGE = [0, 1, 2, 3, 4, 5]  # Number of hidden layers
NEURONS_RANGE = [2, 4, 8, 16, 32, 64]     # Neurons per layer

# Model architecture variants
ARCHITECTURE_VARIANTS = {
    'skip_connections': [None, 'residual', 'dense']
}

# GA parameters (single values for consistency)
GA_PARAMS = {
    'population_size': 100,
    'cxpb': 0.8,
    'mutpb': 0.2,
    'ngen': 100,
    'n_processes': 4,
    'max_time_per_ind': 300,
    'epochs': 20
}

# Model parameters (single values for focused comparison)
MODEL_PARAMS = {
    'activation': 'relu',
    'optimizer': 'adam',
    'lr': 0.001,
    'batch_size': 32
}

# Metrics configuration
METRICS_CONFIG = {
    'tracking': {
        'training': {
            'track_time': True,
            'track_memory': True,
            'track_convergence': True
        },
        'model': {
            'track_parameters': True,
            'track_capacity_utilization': True
        },
        'evaluation': {
            'track_accuracy': True,
            'track_loss': True,
            'track_success_probability': True
        }
    }
}
