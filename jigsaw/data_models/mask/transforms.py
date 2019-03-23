import os
from pathlib import Path
from queue import Queue
from threading import Thread
import pandas as pd

from jigsaw.data_models import mask


def load_labels(skip_background=True):
    """Loads a list of labels from all label map CSVs
        skip_background (bool, optional): Defaults to True. Whether or not to
            skip the label class "Background"
    
    Returns:
        list: a sorted list of label names (unique) across all images
    """
    label_set = set()
    cwd = Path.cwd()
    data_dir = cwd / 'data'

    os.chdir(data_dir)
    for file in os.scandir():
        if not file.name.endswith("_labels.csv"):
            continue
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


def perform_transforms(transforms, image_ids, num_threads=20):
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
            labeled_image_mask = mask.model.LabeledImageMask.construct(
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


class Transform:
    """Stores basic information regarding an image transformation

    Attributes:
        transform_type (str): the type of transform that this is: "rename" or
            "merge"
        original (str/list): the original label that should be renamed, or the
            list of labels that should be merged
        new (str): the output label name after a rename or merge
    """

    def __init__(self, transform_type, original, new):
        self.transform_type = transform_type
        self.original = original
        self.new = new

    def perform_on_image(self, labeled_image_mask):
        """Performs this transformation on a given LabeledImageMask object
        
        Args:
            labeled_image_mask (LabeledImageMask): an object representative of
                a semantically-labeled image
        """
        if self.transform_type == "rename":
            labeled_image_mask.rename_label(self.original, self.new)
        if self.transform_type == "merge":
            labeled_image_mask.merge_labels(self.original, self.new)
