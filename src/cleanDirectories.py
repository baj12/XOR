import os
import shutil


def clean_directories(directories):
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"Removed directory: {directory}")
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    dirs_to_clean = ['logs', 'results', 'models', 'logs/fit', 'plots']
    clean_directories(dirs_to_clean)
