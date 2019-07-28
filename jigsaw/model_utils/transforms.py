import os
import pandas as pd
from pathlib import Path
from queue import Queue
from threading import Thread
from halo import Halo
from colorama import init, Fore
from jigsaw.model_utils.types import Transform
from jigsaw.cli_utils import (user_confirms, user_input, user_selection,
                              FilenameValidator, IntegerValidator, Spinner)

def load_labels(skip_background=True, label_dir=None):
    """Loads a list of labels from all label map CSVs
        skip_background (bool, optional): Defaults to True. Whether or not to
            skip the label class "Background"
   
    Returns:
        list: a sorted list of label names (unique) across all images
    """
    label_set = set()
    cwd = Path.cwd()
    if label_dir is None:
        data_dir = cwd / 'data'
    else:
        data_dir = label_dir

    os.chdir(data_dir)
    for file in os.scandir():
        if file.name.startswith("labels_") and file.name.endswith('.csv'):
            contents = pd.read_csv(file.name)
            labels = contents["label"].tolist()
            for label in labels:
                label_set.add(label)
    if skip_background:
        try:
            label_set.remove("Background")
        except KeyError:
            pass
    os.chdir(cwd)

    return sorted(list(label_set))

def perform_transforms(transforms, image_class, image_ids, num_threads=20):
    """Perform a set of image transformations on a set of images

    This method exists to perform all transforms at once for any given image
    to limit the cost of I/O. It will perform transformations in the order in
    which they exist in the transforms list to avoid attempting to rename/merge
    labels out-of-order. This method uses multithreading to speed up the
    process of loading images/masks/labels as objects, operating upon them,
    and saving the results to disk.
    
    Args:
        transforms (list): a list of Transform objects to be performed in order
            on all images in the image_ids list
        image_ids (list): a list of the image IDs for those images that should
            be transformed
        num_threads (int, optional): Defaults to 20. The number of threads that
            should be used for concurrent transforms.
    """

    # pulls image_ids from a queue, loads the relevant object, performs all
    # transforms in order, and saves the results to the disk
    # NOTE: this is the function being performed concurrently
    def transform(queue):
        while True:
            image_id = queue.get()
            if image_id is None:
                break
            labeled_image_mask = image_class.construct(
                image_id=image_id)
            for transform in transforms:
                transform.perform_on_image(labeled_image_mask)
            queue.task_done()

    # create a queue for images that need to be transformed
    # and spawn threads to perform transforms concurrently
    image_id_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=transform, args=(image_id_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)
    for image_id in image_ids:
        image_id_queue.put(image_id)

    # gracefully finish all threaded processes
    image_id_queue.join()
    for _ in range(num_threads):
        image_id_queue.put(None)
    for worker in workers:
        worker.join()

def default_perform_transforms(image_ids, image_class, label_dir=None, **kwargs):
    transform_list = []

    # ask the user if they would like to perform transforms
    # if yes, enter a loop that supplies transform options
    # if no, skip
    if user_confirms(
            message="Would you like to perform any data transformations?",
            default=False):
        spinner = Spinner(
            text="Loading image labels...", text_color="magenta")
        spinner.start()
        labels = load_labels(label_dir=label_dir)
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
                        new if label == original else label
                        for label in labels
                    ]
                    transform_list.append(
                        Transform(
                            transform_type="rename",
                            original=original,
                            new=new))

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
                            transform_type="merge",
                            original=originals,
                            new=new))

                    if not user_confirms(
                            message="Would you like to continue merging?",
                            default=False):
                        break

        spinner = Spinner(
            text="Performing transformations...", text_color="magenta")
        spinner.start()
        perform_transforms(transform_list, image_class, image_ids=image_ids)
        spinner.succeed(text=spinner.text + "Complete.")

    # collect metadata on transforms
    transform_metadata = []
    for transform in transform_list:
        transform_metadata.append({
            "type": transform.transform_type,
            "original": transform.original,
            "new": transform.new
        })

    return transform_metadata
