def get_process_memory():
    """Get current process memory usage."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def calculate_model_capacity(model):
    """Calculate model capacity utilization."""
    trainable_params = sum([
        np.prod(v.get_shape().as_list())
        for v in model.trainable_variables
    ])
    return trainable_params


class MetricsTracker:
    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(list)

    def track(self, metric_name, value):
        if self.config.metrics['tracking'].get(metric_name, False):
            self.metrics[metric_name].append(value)

    def save_metrics(self, filepath):
        with open(filepath, 'w') as f:
            yaml.dump(dict(self.metrics), f)
