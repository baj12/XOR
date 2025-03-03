#!/usr/bin/env python
import yaml
import glob

def fix_gpu_setting(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure use_gpu is in experiment section
    if 'experiment' not in config:
        config['experiment'] = {}
    
    config['experiment']['use_gpu'] = True
    
    with open(file_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Updated {file_path}")

def main():
    for yaml_file in glob.glob("config/yaml/*.yaml"):
        fix_gpu_setting(yaml_file)

if __name__ == "__main__":
    main()

