import argparse
import logging
import os
import shutil
from contextlib import contextmanager

# Configure logging to include PID
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@contextmanager
def managed_pool():
    # ...existing code...
    try:
        yield
    finally:
        # ...existing code...
        pass


def move_files(destination_name):
    """
    Moves all files from 'logs', 'results', 'models', 'logs/fit' to 'savedResults/<destination_name>'
    while preserving the original directory structure.

    Args:
        destination_name (str): Name of the destination directory under 'savedResults'.
    """
    source_dirs = ['logs', 'results', 'models', 'logs/fit', 'plots']
    destination_dir = os.path.join('savedResults', destination_name)

    os.makedirs(destination_dir, exist_ok=True)
    logger.debug(f"Created destination directory: {destination_dir}")

    for src in source_dirs:
        if os.path.exists(src):
            for root, dirs, files in os.walk(src):
                # Compute the relative path from the source directory
                relative_path = os.path.relpath(root, src)
                # Construct the corresponding destination path
                dest_path = os.path.join(destination_dir, src, relative_path)
                os.makedirs(dest_path, exist_ok=True)
                logger.debug(f"Created directory: {dest_path}")

                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_path, file)
                    try:
                        shutil.move(src_file, dest_file)
                        logger.debug(f"Moved file {src_file} to {dest_file}")
                    except Exception as e:
                        logger.error(
                            f"Failed to move {src_file} to {dest_file}: {e}")
        else:
            logger.warning(f"Source directory does not exist: {src}")

    logger.info(f"All files moved to {destination_dir}")


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Move files from 'logs', 'results', 'models', 'logs/fit' to a specified directory under 'savedResults' while preserving directory structure."
    )
    parser.add_argument(
        'destination_name', type=str,
        help='Name of the destination directory under savedResults.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    move_files(args.destination_name)


if __name__ == "__main__":
    main()
