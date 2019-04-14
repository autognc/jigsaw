#!/usr/bin/env python3
from __future__ import print_function, unicode_literals

import os
import random
import shutil
import time
import numpy as np
random.seed(42)
np.random.seed(42)

from pathlib import Path

from colorama import init, Fore

from jigsaw.cli_utils import (list_to_choices, FilenameValidator,
                              IntegerValidator, DirectoryPathValidator,
                              user_selection, user_input, user_confirms,
                              set_proper_cwd, Spinner)
from jigsaw.data_interface import load_models
from jigsaw.filtering import load_metadata, and_filter, or_filter, join_sets
from jigsaw.io_utils import (
    download_image_data_from_s3, download_json_metadata_from_s3,
    load_BBoxLabeledImages, load_LabeledImageMasks, upload_dataset)
from jigsaw.transforms import load_labels, Transform, perform_transforms
from jigsaw.write_dataset import write_dataset, write_metadata, write_label_map
from jigsaw.models.mask.model import LabeledImageMask

init()

print(Fore.GREEN + "Welcome to Jigsaw!")

set_proper_cwd()

cwd = Path.cwd()
data_dir = cwd / 'data'
try:
    os.makedirs(data_dir)
except FileExistsError:
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)

data_models = load_models()
training_types = [model.training_type for model in data_models]

# ask the user which type of training should be performed
training_type = user_selection(
    message="Which type of training would you like to prepare for?",
    choices=training_types,
    selection_type="list")

model = data_models[training_types.index(training_type)]

data_origin = user_selection(
    message="Would you like to use local data or download data from S3?",
    choices=["Local", "S3"],
    selection_type="list")

if data_origin == "Local":
    data_path = user_input(
        message="Enter the filepath at which the data is located:",
        default=str(Path.home().absolute()),
        validator=DirectoryPathValidator)
    image_ids, filter_metadata = model.filter_and_load(
        data_source=data_origin, data_filepath=data_path)

elif data_origin == "S3":
    default = ""
    default = os.getenv('LABELED_BUCKET_NAME', default)
    bucket = user_input(
        message="Which bucket would you like to download from?",
        default=default)
    image_ids, filter_metadata = model.filter_and_load(
        data_source=data_origin, bucket=bucket)

try:
    transform_metadata = model.transform(image_ids)
except NotImplementedError:
    pass

dataset_name = user_input(
    message="What would you like to name this dataset?",
    validator=FilenameValidator)

k_folds_specified = user_input(
    message="How many folds would you like the dataset to have?",
    validator=IntegerValidator,
    default="5")

comments = user_input("Add any notes or comments about this dataset here:")
user = user_input("Please enter your first and last name:")

spinner = Spinner(text="Writing out dataset locally...", text_color="magenta")
spinner.start()

labeled_images = model.construct_all(image_ids)
write_dataset(
    list(labeled_images.values()),
    custom_dataset_name=dataset_name,
    num_folds=k_folds_specified)

# write out metadata
write_metadata(
    name=dataset_name,
    user=user,
    comments=comments,
    training_type=training_type,
    image_ids=image_ids,
    filters=filter_metadata,
    transforms=transform_metadata)
try:
    model.write_additional_files(dataset_name)
except NotImplementedError:
    pass

spinner.succeed(text=spinner.text + "Complete.")

if user_confirms(message="Would you like to upload the dataset to S3?"):
    default = ""
    default = os.getenv('DATASETS_BUCKET_NAME', default)
    bucket = user_input(
        message="Which bucket would you like to upload to?", default=default)
    spinner = Spinner(text="Uploading dataset to S3...", text_color="magenta")
    spinner.start()
    upload_dataset(
        bucket_name=bucket, directory=Path.cwd() / 'dataset' / dataset_name)
    spinner.succeed(text=spinner.text + "Complete.")

if (dataset_name != '') and user_confirms(
        message="Would you like to delete your " + dataset_name + " dataset?"):
    dataset_path = Path.cwd() / 'dataset' / dataset_name

    spinner = Spinner(
        text="Deleting " + dataset_name + " dataset...", text_color="magenta")
    spinner.start()
    time.sleep(3)
    shutil.rmtree(dataset_path)
    spinner.succeed(text=spinner.text + "Complete.")
