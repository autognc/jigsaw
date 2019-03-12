"""
This file is used to take a list of objects and build
the dataset expected for training data.

Dataset Structure:

/dataset/ <-- *containing folder for all runs*

    /(user defined dataset name for run)/
        /test/
            images + truth data
        /dev/
            /standard/ <-- *replica of fold_0*
            /fold_i/
                /tf/
                    train.record
                    test.record
                /validation/
                    images + truth data

            ...

            /fold_n/
                ...
"""

# --- Imports.
import os, shutil, time, json
import numpy as np

from datetime import datetime
from random import shuffle
from pathlib import Path

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from sklearn.model_selection import KFold

from mask import LabeledImageMask
from bounding_box import BBoxLabeledImage

# --- Variables.
default_test_percentage = 0.2
default_num_folds = 5
default_out_path = Path.cwd()


def delete_dir(path):
    if os.path.isdir(path):
        print("Deleting contents of", path,
              "you've got five seconds to cancel this.")
        time.sleep(5)

        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
                raise Exception(
                    "Directory deletion failed. Maybe you have a file explorer/terminal open in this dir?"
                )

        print("Done deleting contents of", path)


def split_data(obj_list, test_percent=0.2):
    if len(obj_list) == 0:
        raise Exception("Empty object list passed.")

    shuffle(obj_list)
    index_to_split_on = max(1, int(len(obj_list) * test_percent))

    test = obj_list[:index_to_split_on]
    dev = obj_list[index_to_split_on:]

    return (test, dev)


def divide_into_folds(obj_list, num_folds=default_num_folds):
    key_to_object = dict()
    key_list = []

    for key, object_item in enumerate(obj_list):
        key_to_object[key] = object_item
        key_list.append(key)

    key_list = np.array(key_list)

    # NOTE: random_state variable sets seed.
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=1)

    divided_list_of_lists = []
    for train_index, validation_index in kfold.split(key_list):
        train_key_list = list(key_list[train_index])
        validation_key_list = list(key_list[validation_index])

        train_data = []
        validation_data = []

        for key in train_key_list:
            train_data.append(key_to_object[key])

        for key in validation_key_list:
            validation_data.append(key_to_object[key])

        divided_list_of_lists.append((train_data, validation_data))

    return divided_list_of_lists


def write_out_fold(path, fold_data):
    record_path = path / 'tf'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    train_subset = fold_data[0]
    validation_subset = fold_data[1]

    test_record_data, train_record_data = split_data(train_subset)

    write_related_data(validation_subset, path / 'validation')
    write_out_tf_examples(train_record_data, record_path / 'train.record')
    write_out_tf_examples(test_record_data, record_path / 'test.record')


def write_out_tf_examples(objects, path):
    num_shards = (len(objects) // 1000) + 1

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, str(path.absolute()), num_shards)

        for index, object_item in enumerate(objects):
            tf_example = object_item.convert_to_tf_example()
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(
                tf_example.SerializeToString())


# Only BBLI has a convert_to_dict method for now, hence type check.
def write_obj_to_json(obj: BBoxLabeledImage, path):
    obj_converted_to_dict = obj.convert_to_dict()

    with open(path, 'w') as outfile:
        json.dump(obj_converted_to_dict, outfile)


def write_related_data(objects, path):
    if not os.path.exists(path):
        os.makedirs(path)

    sem_folder_ext_pairs = [('images', '.jpg'), ('json', '_meta.json'),
                            ('labels', '_labels.csv'), ('masks', '_mask.png')]
    bbli_folder_ext_pairs = [('images', '.jpg'), ('json', '_meta.json'),
                             ('', '_boxes.json')]

    data_path = default_out_path / 'data'

    for obj in objects:
        if isinstance(obj, LabeledImageMask):
            for (location, ext) in sem_folder_ext_pairs:
                name = obj.image_id + ext

                orig_path = data_path / location / name
                new_path = path / name

                shutil.copyfile(orig_path, new_path)

        elif isinstance(obj, BBoxLabeledImage):
            for (location, ext) in bbli_folder_ext_pairs:
                if location != '':
                    name = obj.image_id + ext

                    orig_path = data_path / location / name
                    new_path = path / name

                    shutil.copyfile(orig_path, new_path)
                else:
                    write_obj_to_json(obj,
                                      path / str(obj.image_id + '_boxes.json'))

        else:
            raise Exception("Hmmm, don't recognize this object type.")


def write_dataset(obj_list,
                  test_percent=default_test_percentage,
                  num_folds=default_num_folds,
                  out_dir: Path = default_out_path,
                  custom_dataset_name='dataset'):

    dataset_path = out_dir / 'dataset' / custom_dataset_name
    delete_dir(dataset_path)

    test_subset, dev_subset = split_data(obj_list, test_percent)

    # Test subset.
    write_related_data(test_subset, dataset_path / 'test')

    # Fold subsets.
    folds = divide_into_folds(dev_subset)
    dev_path = dataset_path / 'dev'

    for fold_num, fold in enumerate(folds):

        # The special standard set.
        if fold_num == 0:
            fold_path = dev_path / 'standard'
            write_out_fold(fold_path, fold)

        fold_path = dev_path / str('fold_' + str(fold_num))
        write_out_fold(fold_path, fold)


def write_metadata(
        name,
        user,
        comments,
        training_type,
        image_ids,
        filters,
        transforms,
        out_dir: Path = default_out_path,
):
    """Writes out a metadata file in JSON format

    Args:
        name (str): the name of the dataset
        comments (str): comments or notes supplied by the user regarding the
            dataset produced by this tool
        training_type (str): the training type selected by the user
        image_ids (list): a list of image IDs that ended up in the final
            dataset (either dev or test)
        filters (dict): a dictionary representing filter metadata
        transforms (dict): a dictionary representing transform metadata
        out_dir (Path, optional): Defaults to default_out_path.
    """
    dataset_path = out_dir / 'dataset' / name
    metadata_filepath = dataset_path / 'metadata.json'

    metadata = {}
    metadata["name"] = name
    metadata["date_created"] = datetime.utcnow().isoformat() + "Z"
    metadata["created_by"] = user
    metadata["comments"] = comments
    metadata["training_type"] = training_type
    metadata["image_ids"] = image_ids
    metadata["filters"] = filters
    metadata["transforms"] = transforms
    with open(metadata_filepath, 'w') as outfile:
        json.dump(metadata, outfile)


def write_label_map(name, out_dir: Path = default_out_path):
    """Writes out the TensorFlow Object Detection Label Map
    
    Args:
        name (str): the name of the dataset
        out_dir (Path, optional): Defaults to default_out_path.
    """
    dataset_path = out_dir / 'dataset' / name
    label_map_filepath = dataset_path / 'label_map.pbtxt'
    label_map = []
    for label_name, label_int in LabeledImageMask.label_to_int_dict.items():
        label_info = "\n".join([
            "item {", "  id: {id}".format(id=label_int),
            "  name: '{name}'".format(name=label_name), "}"
        ])
        label_map.append(label_info)
    with open(label_map_filepath, 'w') as outfile:
        outfile.write("\n\n".join(label_map))
