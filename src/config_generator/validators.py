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
    try:
        # Check input dimensions match
        if config['data']['input_dim'] != config['model']['input_dim']:
            raise ValueError("Data and model input dimensions must match")

        # Verify input dimensions account for noise
        expected_input_dim = 2 + config['experiment']['noise_dimensions']
        if config['data']['input_dim'] != expected_input_dim:
            raise ValueError(f"Input dimension should be {expected_input_dim} "
                             f"(2 + {config['experiment']['noise_dimensions']} noise dimensions)")

        # Validate hidden layers and skip connections
        if not config['model']['hidden_layers'] and config['model']['skip_connections']:
            raise ValueError(
                "Skip connections require at least one hidden layer")

        # Additional validations...

        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
