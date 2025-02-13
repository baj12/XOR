def validate_parameters(params):
    """Validate parameter combinations and constraints."""
    validations = {
        'architecture': validate_architecture_params(params),
        'noise': validate_noise_params(params),
        'ga': validate_ga_params(params),
        'metrics': validate_metrics_params(params)
    }
    return all(validations.values())


def validate_architecture_params(params):
    if params['hidden_layers'] == 0 and params['skip_connections']:
        return False
    if params['neurons'] < 2:
        return False
    return True


def validate_noise_params(params):
    if params['noise_dim'] < 0:
        return False
    if not 0 <= params['separation'] <= 1:
        return False
    return True


def validate_ga_params(params):
    # Add GA parameter validation
    return True


def validate_metrics_params(params):
    # Add metrics parameter validation
    return True


def validate_config(config):
    """Validate the complete configuration."""
    # Check required sections
    required_sections = ['experiment', 'data', 'ga', 'model', 'metrics']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")

    # Validate model config
    if not isinstance(config['model']['hidden_layers'], list):
        raise ValueError("hidden_layers must be a list")
    if not isinstance(config['model']['neurons_per_layer'], int):
        raise ValueError("neurons_per_layer must be an integer")
    if config['model']['activation'] not in ['relu', 'tanh']:
        raise ValueError("activation must be 'relu' or 'tanh'")

    # Validate GA config
    if not 0 <= config['ga']['cxpb'] <= 1:
        raise ValueError("crossover probability must be between 0 and 1")
    if not 0 <= config['ga']['mutpb'] <= 1:
        raise ValueError("mutation probability must be between 0 and 1")

    return True
