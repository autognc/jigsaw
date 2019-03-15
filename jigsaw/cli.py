#!/usr/bin/env python3
from __future__ import print_function, unicode_literals

from pathlib import Path

from colorama import init, Fore
from halo import Halo

from cli_utils import (list_to_choices, FilenameValidator, IntegerValidator,
                       user_selection, user_input, user_confirms)
from filtering import load_metadata, and_filter, or_filter, join_sets
from io_utils import (download_image_data_from_s3, load_BBoxLabeledImages,
                      load_LabeledImageMasks, upload_dataset)
from transforms import load_labels, Transform, perform_transforms
from write_dataset import write_dataset, write_metadata, write_label_map

init()

print(Fore.GREEN + "Welcome to Jigsaw!")

# ask the user which type of training should be performed
training_type = user_selection(
    message="Which type of training would you like to prepare for?",
    choices=["Bounding Box", "Semantic Segmentation"],
    selection_type="list")

# load all image tags before asking the user to filter data
spinner = Halo(
    text="Downloading image metadata from S3...", text_color="magenta")
spinner.start()
tags_df = load_metadata()
spinner.succeed(text=spinner.text + "Complete.")

filter_metadata = {"groups": []}
# ask the user if they would like to perform filtering
# if yes, enter a loop that supplies filter options
# if no, skip

if user_confirms(
        message="Would you like to filter out any of the data hosted on S3?",
        default=True):
    sets = {}
    # outer loop to determine how many sets the user will create
    while True:
        subset = tags_df
        this_group_filters = []
        len_subsets = [len(subset)]
        # inner loop to handle filtering for ONE set
        while True:

            # if filters have been applied, display them to the user
            # to help guide their next choice
            if len(this_group_filters) > 0:
                filters_applied = [
                    "   > " + (" " + f["type"] + " ").join(f["tags"]) +
                    " ({} -> {})".format(len_subsets[i], len_subsets[i + 1])
                    for i, f in enumerate(this_group_filters)
                ]
                print(Fore.MAGENTA + "ℹ Filters already applied:\n{}".format(
                    "\n".join(filters_applied)))

            selected_tags = user_selection(
                message=
                "Please select a set of tags with which to apply a filter:",
                choices=list(tags_df),
                selection_type="checkbox")
            filter_type = user_selection(
                message=
                "Which filter would you like to apply to the above set?",
                choices=["AND (intersection)", "OR (union)"],
                selection_type="list")

            if filter_type == "AND (intersection)":
                subset = and_filter(subset, selected_tags)
                this_group_filters.append({
                    "type": "AND",
                    "tags": selected_tags
                })
            elif filter_type == "OR (union)":
                subset = or_filter(subset, selected_tags)
                this_group_filters.append({
                    "type": "OR",
                    "tags": selected_tags
                })
            print(
                Fore.GREEN +
                "ℹ There are {} images that meet the filter criteria selected."
                .format(len(subset)))
            len_subsets.append(len(subset))

            if not user_confirms(
                    message="Would you like to continue filtering this set?",
                    default=False):
                set_name = user_input(
                    message="What would you like to name this set?",
                    validator=FilenameValidator)
                sets[set_name] = subset
                filter_metadata["groups"].append({
                    "name": set_name,
                    "filters": this_group_filters
                })
                break

        if not user_confirms(
                message="Would you like to create more sets via filtering?",
                default=False):
            break

    sets_to_join = []
    for set_name, set_data in sets.items():
        how_many = user_input(
            message='How many images of set "{}" would you like to use? (?/{})'
            .format(set_name, len(set_data)),
            validator=IntegerValidator)
        n = int(how_many)
        sets_to_join.append(set_data.sample(n, replace=False, random_state=42))

        # find the right group within the metadata dict and add the number
        # included to it
        for group in filter_metadata["groups"]:
            if group["name"] == set_name:
                group["number_included"] = n

    results = join_sets(sets_to_join)

else:
    results = tags_df

# download the semantic masks and label maps from S3 for those images that
# have passed through the filter
spinner.text = "Downloading image masks and labels from S3..."
spinner.start()
download_image_data_from_s3(image_ids=results.index.tolist())
spinner.succeed(text=spinner.text + "Complete.")

transform_list = []

# ask the user if they would like to perform transforms
# if yes, enter a loop that supplies transform options
# if no, skip
if user_confirms(
        message="Would you like to perform any data transformations?",
        default=False):

    spinner.text = "Loading image labels..."
    spinner.start()
    labels = load_labels()
    spinner.succeed(text=spinner.text + "Complete.")

    while True:
        transform = user_selection(
            message="Which type of transformation?",
            choices=["Label Renaming", "Label Merging", "Exit"],
            sort_choices=False,
            selection_type="list")

        if transform == "Exit":
            break

        if transform == "Label Renaming":
            while True:
                original = user_selection(
                    message="Which label would you like to rename?",
                    choices=labels,
                    selection_type="list")
                new = user_input(
                    message="What should the new name be?",
                    validator=FilenameValidator)

                labels = [
                    new if label == original else label for label in labels
                ]
                transform_list.append(
                    Transform(
                        transform_type="rename", original=original, new=new))

                if not user_confirms(
                        message="Would you like to continue renaming?",
                        default=False):
                    break

        if transform == "Label Merging":
            while True:
                # insert transforms here
                originals = user_selection(
                    message="Which labels should be merged into one?",
                    choices=labels,
                    selection_type="checkbox")
                new = user_input(
                    message="What should the merged name be?",
                    validator=FilenameValidator)

                for original in originals:
                    labels.remove(original)
                labels.append(new)
                transform_list.append(
                    Transform(
                        transform_type="merge", original=originals, new=new))

                if not user_confirms(
                        message="Would you like to continue merging?",
                        default=False):
                    break

    spinner.text = "Performing transformations..."
    spinner.start()
    perform_transforms(transform_list, image_ids=results.index.tolist())
    spinner.succeed(text=spinner.text + "Complete.")

# collect metadata on transforms
transform_metadata = []
for transform in transform_list:
    transform_metadata.append({
        "type": transform.transform_type,
        "original": transform.original,
        "new": transform.new
    })

dataset_name = user_input(
    message="What would you like to name this dataset?",
    validator=FilenameValidator)

k_folds_specified = user_input(
    message="How many folds would you like the dataset to have?",
    validator=IntegerValidator,
    default="5")

comments = user_input("Add any notes or comments about this dataset here:")
user = user_input("Please enter your first and last name:")

spinner.text = "Writing out dataset locally..."
spinner.start()

if training_type == "Bounding Box":
    bbox_labeled_images = load_BBoxLabeledImages(
        image_ids=results.index.tolist())
    bbox_labeled_images = list(bbox_labeled_images.values())

    write_dataset(
        bbox_labeled_images,
        custom_dataset_name=dataset_name,
        num_folds=k_folds_specified)

elif training_type == "Semantic Segmentation":
    labeled_image_masks = load_LabeledImageMasks(
        image_ids=results.index.tolist())
    write_dataset(
        labeled_image_masks,
        custom_dataset_name=dataset_name,
        num_folds=k_folds_specified)

# write out metadata
write_metadata(
    name=dataset_name,
    user=user,
    comments=comments,
    training_type=training_type,
    image_ids=results.index.tolist(),
    filters=filter_metadata,
    transforms=transform_metadata)

write_label_map(name=dataset_name)
spinner.succeed(text=spinner.text + "Complete.")

#spinner.text = "Uploading dataset to S3..."
#spinner.start()
#upload_dataset(directory=Path.cwd() / 'dataset' / dataset_name)
#spinner.succeed(text=spinner.text + "Complete.")
