#!/usr/bin/env python3
from __future__ import print_function, unicode_literals

import os
import random
import shutil
import numpy as np
random.seed(42)
np.random.seed(42)

from pathlib import Path

from colorama import init, Fore
from halo import Halo

from jigsaw.cli_utils import (list_to_choices, FilenameValidator,
                              IntegerValidator, FilepathValidator,
                              user_selection, user_input, user_confirms)
from jigsaw.filtering import load_metadata, and_filter, or_filter, join_sets
from jigsaw.io_utils import (
    download_image_data_from_s3, download_json_metadata_from_s3,
    load_BBoxLabeledImages, load_LabeledImageMasks, upload_dataset)
from jigsaw.transforms import load_labels, Transform, perform_transforms
from jigsaw.write_dataset import write_dataset, write_metadata, write_label_map
from jigsaw.data_models import mask

init()

print(Fore.GREEN + "Welcome to Jigsaw!")

cwd = Path.cwd()
data_dir = cwd / 'data'
try:
    os.makedirs(data_dir)
except FileExistsError:
    shutil.rmtree(data_dir)
    os.makedirs(data_dir)

# ask the user which type of training should be performed
training_type = user_selection(
    message="Which type of training would you like to prepare for?",
    choices=["Bounding Box", "Semantic Segmentation"],
    selection_type="list")

if training_type == "Semantic Segmentation":
    model = mask.model.LabeledImageMask
else:
    raise NotImplementedError

data_origin = user_selection(
    message="Would you like to use local data or download data from S3?",
    choices=["Local", "S3"],
    selection_type="list")

if data_origin == "Local":
    # TODO: fix validator
    data_path = user_input(
        message="Enter the filepath at which the data is located:",
        default=str(Path.home().absolute()),
        validator=FilepathValidator)
    image_ids, filter_metadata = model.filter_and_load(
        data_source=data_origin, data_filepath=data_path)

elif data_origin == "S3":
    bucket = user_input(
        message="Which bucket would you like to download from?",
        default=os.environ["LABELED_BUCKET_NAME"])
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

spinner = Halo(text="Writing out dataset locally...", text_color="magenta")
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
    bucket = user_input(
        message="Which bucket would you like to upload to?",
        default=os.environ["DATASETS_BUCKET_NAME"])
    spinner = Halo(text="Uploading dataset to S3...", text_color="magenta")
    spinner.start()
    upload_dataset(
        bucket_name=bucket, directory=Path.cwd() / 'dataset' / dataset_name)
    spinner.succeed(text=spinner.text + "Complete.")
