import os.path

import datasets

__all__ = ["get_dataset"]


def get_dataset(
    name: str, config_name = None, split: str = "train", return_gt: bool = False
) -> datasets.Dataset:
    prefix = os.path.dirname(__file__)
    kwargs = {"name": config_name, "split": split, "trust_remote_code": True, "token": True}
    path = os.path.join(prefix, f"{name}")
    if name == "DCI":
        dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
    elif name == "MJHQ":
        dataset = datasets.load_dataset(path, return_gt=return_gt, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    return dataset
