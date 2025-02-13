from datetime import datetime


def generate_filename(params):
    """Generate configuration filenames."""
    timestamp = datetime.now().strftime("%Y%m%d")
    return (f"config_{timestamp}"
            f"_h{params['hidden_layers']}"
            f"_n{params['neurons']}"
            f"_noise{params['noise_dim']}"
            f"_sep{params['separation']:.2f}"
            f"_skip{params['skip_connections']}"
            f"_{params['id']:03d}.yaml")


def setup_results_structure(config):
    """Create results directory structure."""
    experiment_id = config['experiment']['id']
    dirs = {
        'models': f'results/{experiment_id}/models',
        'metrics': f'results/{experiment_id}/metrics',
        'plots': f'results/{experiment_id}/plots',
        'logs': f'results/{experiment_id}/logs',
        'analysis': f'results/{experiment_id}/analysis'
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs
