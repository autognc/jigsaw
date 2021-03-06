"""
This file is used to take a list of objects and build
the dataset expected for training data.

Dataset Structure:

/dataset/ <-- *containing folder for all runs*

    /(user defined dataset name for run)/
        /test/
            images + truth data
        /dev/
            /complete/ <-- *all of dev set, no validation set*
            /standard/ <-- *standard 80/20 split for records/validation*
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

import os, shutil, time, json
import numpy as np

from datetime import datetime
from random import shuffle
from pathlib import Path

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
from sklearn.model_selection import KFold

def delete_dir(path):
    """Deletes all files and folders in specified directory
    
    Args:
        path (Path object): target directory
    """
    if os.path.isdir(path):
        print('\n', "Deleting contents of", path,
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

        print('\n', "Done deleting contents of", path)


def split_data(obj_list, test_percent=0.2):
    """Splits obj_list into test/dev sets
    
    Args:
        obj_list (list): list of objects to divide into test/dev
        test_percent (int): percentage of objects in the test set
    
    Returns:
        tuple of two lists. The first list is test, second dev
    """
    if len(obj_list) == 0:
        raise Exception("Empty object list passed.")
    if len(obj_list) == 1:
        raise Exception(
            "Object list of length 1 passed. Can't build test and dev set with this."
        )

    shuffle(obj_list)
    index_to_split_on = max(1, int(len(obj_list) * test_percent))

    test = obj_list[:index_to_split_on]
    dev = obj_list[index_to_split_on:]

    return (test, dev)


def divide_into_folds(obj_list, num_folds=5):
    """Splits obj_list into num_folds folds using sklearn's KFold
    
    Args:
        obj_list (list): list of objects to divide into test/dev
        num_folds (int): number of folds to produce
    
    Returns:
        tuple of two lists. The first list is train, second validation
    """
    key_to_object = dict()
    key_list = []

    for key, object_item in enumerate(obj_list):
        key_to_object[key] = object_item
        key_list.append(key)

    key_list = np.array(key_list)

    # NOTE: random_state variable sets seed.
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)

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


def write_out_fold(path, fold_data, is_standard=False):
    """Writes out the fold_data to specified path
    
    Args:
        path (Path): directory to write folds out to
        fold_data (list): list of tuples, each tuple is (train, validation) sets
    """
    record_path = path / 'train'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    # Making it so for standard set, validation set size isn't reliant on KFold size.
    if is_standard:
        fold_data = split_data(fold_data[0] + fold_data[1])
        fold_data = (fold_data[1], fold_data[0])

    train_subset = fold_data[0]
    validation_subset = fold_data[1]

    test_record_data, train_record_data = split_data(train_subset)

    write_related_data(validation_subset, path / 'dev')
    
    # training
    write_out_tf_examples(train_record_data, record_path / 'train.record')
    write_out_tf_examples(test_record_data, record_path / 'test.record')


def write_out_complete_set(path, data):
    """Writes out the special complete set into 'dev'. No validation data.
    
    Args:
        path (Path): directory to write complete set out to
        data (list): objects to write to the complete set
    """
    record_path = path / 'train'
    if not os.path.exists(record_path):
        os.makedirs(record_path)

    test_record_data, train_record_data = split_data(data)

    write_out_tf_examples(train_record_data, record_path / 'train.record')
    write_out_tf_examples(test_record_data, record_path / 'test.record')


def write_out_tf_examples(objects, path):
    """Writes out list of objects out as a single tf_example
    
    Args:
        objects (list): list of objects to put into the tf_example 
        path (Path): directory to write this tf_example to, encompassing the name
    """
    num_shards = (len(objects) // 1000) + 1
    
    with open(str(path) + '.numexamples', 'w') as output:
        output.write(str(len(objects)))

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, str(path.absolute()), num_shards)

        for index, object_item in enumerate(objects):
            tf_example = object_item.export_as_TFExample()
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(
                tf_example.SerializeToString())

def write_related_data(objects, path):
    if not os.path.exists(path):
        os.makedirs(path)

    for obj in objects:
        obj.copy_associated_files(path)


def write_dataset(obj_list,
                  test_percent=0.2,
                  num_folds=5,
                  custom_dataset_name='dataset'):
    """Main driver for this file
    
    Args:
        obj_list (list): objects that will be transformed into usable dataset
        test_percent (float): percent of data that'll be test data
        num_folds (int): number of folds to write
        out_dir (Path): directory to write this dataset to
        custom_dataset_name (str): name of the dataset's containing folder
    """
    out_dir = Path.cwd()
    dataset_path = out_dir / 'dataset' / custom_dataset_name
    print(dataset_path)
    delete_dir(dataset_path)

    test_subset, dev_subset = split_data(obj_list, test_percent)

    # Test subset.
    write_related_data(test_subset, dataset_path / 'test')

    dev_path = dataset_path / 'splits'

    # standard_path = dev_path / 'standard'
    # write_out_fold(standard_path, fold, is_standard=True)

    complete_path = dev_path / 'complete'
    write_out_complete_set(complete_path, dev_subset)



def write_metadata(
        name,
        user,
        comments,
        training_type,
        image_ids,
        filters,
        transforms
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
        out_dir (Path, optional): Defaults to Path.cwd().
    """
    out_dir = Path.cwd()
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


def write_label_map(name, out_dir: Path = Path.cwd()):
    """Writes out the TensorFlow Object Detection Label Map
    
    Args:
        name (str): the name of the dataset
        out_dir (Path, optional): Defaults to Path.cwd().
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
