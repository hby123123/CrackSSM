import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DEFAULT_CKPT_ROOT = os.path.join(PROJECT_ROOT, "checkpoints")
DEFAULT_INPUT_SIZE = [448, 448]


def get_data_root(override=None):
    if override is None:
        return DEFAULT_DATA_ROOT
    if os.path.isabs(override):
        return override
    return os.path.join(PROJECT_ROOT, override)


def get_ckpt_dir(experiment_name, override_root=None):
    root = override_root or DEFAULT_CKPT_ROOT
    return os.path.join(root, experiment_name)
