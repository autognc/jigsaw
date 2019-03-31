import os
from pathlib import Path


def sample_data_path():
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    return Path(parent_dir) / "sample_data"
