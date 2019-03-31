import importlib
import os
import shutil
import yaml

from pathlib import Path
from types import ModuleType

from jigsaw.write_dataset import (write_dataset, write_metadata)
from jigsaw import models


def test_models():
    config = Path(os.path.dirname(
        os.path.realpath(__file__))) / 'test_config.yml'

    with open(config) as f:
        data_models = yaml.safe_load(f)

    for data_model in data_models:
        module = importlib.import_module(data_model["parent_module"])
        model = getattr(module, data_model["model_class"])
        this_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        data_path = this_dir.joinpath(
            data_model["sample_data_relpath"]).resolve()
        _test_model(model, data_path)


def _test_model(model, data_path):
    image_ids, filter_metadata = model.filter_and_load(
        data_source="Local", data_filepath=data_path)

    try:
        transform_metadata = model.transform(image_ids)
    except NotImplementedError:
        pass

    labeled_images = model.construct_all(image_ids)

    dataset_name = "test"
    write_dataset(
        list(labeled_images.values()),
        custom_dataset_name=dataset_name,
        num_folds=5)
    write_metadata(
        name=dataset_name,
        user="Test User",
        comments="This is a test.",
        training_type=model.training_type,
        image_ids=image_ids,
        filters=filter_metadata,
        transforms=transform_metadata)
    try:
        model.write_additional_files(dataset_name)
    except NotImplementedError:
        pass

    shutil.rmtree(Path.cwd() / "data")
    shutil.rmtree(Path.cwd() / "dataset")
