import importlib
import os
import shutil
import yaml

from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from threading import Thread

from jigsaw import models


class LabeledImage(ABC):
    def __init__(self, image_id):
        self.image_id = image_id

    @classmethod
    @abstractmethod
    def construct(cls, image_id, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def filter_and_load(cls, data_source, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transform(cls, image_ids, **kwargs):
        raise NotImplementedError

    # file extensions of any file relevant to this model
    @property
    @classmethod
    @abstractmethod
    def associated_files(cls):
        raise NotImplementedError
    
    # prefixes of any file actually needed to validation/testing data from this model
    @property
    @classmethod
    @abstractmethod
    def related_data_prefixes(cls):
        raise NotImplementedError

    @property
    @classmethod
    def training_type(cls):
        raise NotImplementedError

    @abstractmethod
    def export_as_TFExample(self, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def write_additional_files(cls, dataset_name, **kwargs):
        raise NotImplementedError

    @classmethod
    def construct_all(cls, image_ids, num_threads=20):
        """Loads a set of LabeledImage objects from the filesystem

        NOTE: this is done concurrently to limit I/O costs.
        
        Args:
            image_ids (list): the list of image IDs that should be loaded
            num_threads (int, optional): Defaults to 20. The number of threads that
                should be used for concurrent loading.
        
        Returns:
            dict: a dict where the keys are image IDs and the values are the
                object for each image ID
        """
        labeled_images = {}

        # pulls image_ids from a queue, loads the relevant object
        # NOTE: this is the function being performed concurrently
        def worker_load_func(queue):
            while True:
                image_id = queue.get()
                if image_id is None:
                    break
                labeled_images[image_id] = cls.construct(image_id)
                queue.task_done()

        # create a queue for images that need to be loaded
        image_id_queue = Queue(maxsize=0)
        workers = []
        for worker in range(num_threads):
            worker = Thread(target=worker_load_func, args=(image_id_queue, ))
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

        return labeled_images

    def copy_associated_files(self, destination, **kwargs):
        data_dir = Path.cwd() / "data"
        for suffix in self.associated_files.values():
            for prefix in self.related_data_prefixes.values():
                filepath = data_dir / f'{prefix}{self.image_id}{suffix}'
                if filepath.exists():
                    shutil.copy(
                        str(filepath.absolute()), str(destination.absolute()))


def load_models():
    parent_dir = os.path.dirname(os.path.realpath(__file__))
    data_models_yml = Path(parent_dir) / "data_models.yml"
    with open(data_models_yml) as f:
        data_models = yaml.safe_load(f)
    model_list = []
    for data_model in data_models:
        module = importlib.import_module(data_model["parent_module"])
        model_list.append(getattr(module, data_model["model_class"]))
    return model_list