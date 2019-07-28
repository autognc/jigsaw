import os
import requests
import shutil

from pathlib import Path
from queue import Queue
from threading import Thread
from PIL import Image
from io import BytesIO

import boto3
import cv2
import numpy as np

from jigsaw.mask import LabeledImageMask
from jigsaw.bounding_box import BBoxLabeledImage


def download_image_and_save(image_url, destination):
    """Downloads an image stored remotely and saves it locally
    
    Args:
        image_url (str): the URL of the image
        destination (str): the local filepath at which to save the image
    """
    response = requests.get(image_url, stream=True)
    with open(destination, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


def load_remote_image(image_url):
    """Loads a remotely stored image into memory as an OpenCV/Numpy array
    
    Args:
        image_url (str): the URL of the image
    
    Returns:
        numpy ndarray: the image in OpenCV format (a [rows, cols, 3] BGR numpy
            array)
    """
    response = requests.get(image_url, stream=True)
    img = Image.open(BytesIO(response.content))
    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return image


def copy_data_locally(source_dir,
                      condition_func=lambda filename: True,
                      num_threads=20):
    """Copies data from a local source into Jigsaw given some condition
    
    Args:
        source_dir (str or pathlib `Path`): the source directory from which
            data should be copied
        condition_func (function, optional): a function that uses the filename
            to determine whether data should be copied. This allows for only
            copying portions of the source data if desired. Defaults to True
            for all filenames.
        num_threads (int, optional): Defaults to 20. Number of threads
            performing concurrent copies.
    """
    # convert the source directory from a string to a pathlib Path if necessary
    if isinstance(source_dir, str):
        source_dir = Path(source_dir)

    cwd = Path.cwd()
    data_dir = cwd / 'data'
    try:
        os.makedirs(data_dir)
    except FileExistsError:
        pass

    def copy_object(queue):
        while True:
            filepath = queue.get()
            if filepath is None:
                break
            shutil.copy(filepath, data_dir.absolute())
            queue.task_done()

    # create a queue for objects that need to be copied
    # and spawn threads to copy them concurrently
    copy_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=copy_object, args=(copy_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    for file in source_dir.iterdir():
        if condition_func(file.name) and not (data_dir / file.name).exists():
            copy_queue.put(file.absolute())

    copy_queue.join()
    for _ in range(num_threads):
        copy_queue.put(None)
    for worker in workers:
        worker.join()


def download_data_from_s3(bucket_name,
                          filter_vals=[''],
                          condition_func=lambda filename: True,
                          num_threads=20):
    """Downloads data from S3 into Jigsaw given some condition
    
    Args:
        bucket_name (str): the name of the S3 bucket to download from
        condition_func ([type], optional): a function that uses the filename
            to determine whether data should be downloaded. This allows for
            only downloading portions of the source data if desired. Defaults
            to True for all filenames.
        num_threads (int, optional): Defaults to 20. Number of threads
            performing concurrent downloads.
    """

    # simple method for threads to pull from a queue and download JSON files
    def download_object(queue):
        while True:
            obj = queue.get()
            if obj is None:
                break
            obj.Object().download_file(obj.key.split("/")[-1])
            # TODO: test for files with prefixes here
            queue.task_done()

    cwd = Path.cwd()
    data_dir = cwd / 'data'
    # try:
    os.makedirs(data_dir, exist_ok=True)
    # except FileExistsError:
        # pass
    os.chdir(data_dir)

    # create a queue for objects that need to be copied
    # and spawn threads to copy them concurrently
    download_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=download_object, args=(download_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    # loop through the files in the bucket and add them to the queue if they
    # satisfy the condition_func
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for prefix in filter_vals:
        for obj in bucket.objects.filter(Prefix=prefix):
            filename = obj.key.split("/")[-1]
            if condition_func(filename) and not (data_dir / filename).exists():
                download_queue.put(obj)

    # wait for the queue to be empty, then join all threads
    download_queue.join()
    for _ in range(num_threads):
        download_queue.put(None)
    for worker in workers:
        worker.join()

    os.chdir(cwd)


def download_json_metadata_from_s3(bucket_name, prefix="", num_threads=20):
    """Downloads JSON metadata files for a set of images with a given S3 prefix

    This function downloads the JSON files and puts them in a folder called
    "json" in the current working directory.

    Args:
        bucket_name (str): the name of the bucket that contains the JSON
            metadata.
        prefix (str): Defaults to "". The S3 prefix (conceptually
            similar to a folder) with which you want to filter images
        num_threads (int): Defaults to 20. The number of threads that should be
            used for concurrent downloads.
    """

    # simple method for threads to pull from a queue and download JSON files
    def download_object(queue):
        while True:
            obj = queue.get()
            if obj is None:
                break
            obj.Object().download_file(obj.key.replace(prefix, ''))
            queue.task_done()

    # create a directory to store downloaded metadata
    cwd = Path.cwd()
    data_dir = cwd / 'data'
    json_dir = data_dir / 'json'
    # try:
    os.makedirs(json_dir, exist_ok=True)
    # except FileExistsError:
    #     shutil.rmtree(json_dir)
    #     os.makedirs(json_dir)
    os.chdir(json_dir)

    # create a queue for objects that need to be downloaded
    # and spawn threads to download them concurrently
    download_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=download_object, args=(download_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    # loop through the files in the bucket and filter for JSON metadata
    # files for only labeled images; add them to the queue
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("meta.json"):
            download_queue.put(obj)

    # wait for the queue to be empty, then join all threads
    download_queue.join()
    for _ in range(num_threads):
        download_queue.put(None)
    for worker in workers:
        worker.join()

    os.chdir(cwd)


def download_image_data_from_s3(image_ids, prefix="", num_threads=20):
    """Downloads semantic masks, label maps, and original images from S3

    This function downloads the semantic mask PNG files and puts them in a
    folder called "masks" in the current working directory, and it does the
    same for label map CSVs but puts them in a folder called "labels".

    Args:
        image_ids (list): a list of unique IDs (str) for images whose label
            maps and masks should be downloaded
        prefix (str): Defaults to "". The S3 prefix (conceptually
            similar to a folder) with which you want to filter images
        num_threads (int): Defaults to 20. The number of threads that should be
            used for concurrent downloads.
    """

    # simple method for threads to pull from a queue and download files
    def download_object(queue):
        while True:
            obj = queue.get()
            if obj is None:
                break
            obj.Object().download_file(obj.key.lstrip(prefix))
            queue.task_done()

    # create a directory to store downloaded metadata

    cwd = Path.cwd()
    data_dir = cwd / 'data'
    masks_dir = data_dir / 'masks'
    try:
        os.mkdir(masks_dir)
    except FileExistsError:
        shutil.rmtree(masks_dir)
        os.mkdir(masks_dir)
    os.chdir(masks_dir)

    # create a queue for objects that need to be downloaded
    # and spawn threads to download them concurrently
    download_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=download_object, args=(download_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    # loop through the files in the bucket and filter for JSON metadata
    # files for only labeled images; add them to the queue
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(os.environ["LABELED_BUCKET_NAME"])
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("_mask.png"):
            image_id = obj.key.replace(prefix,'').replace("_mask.png", '')
            if image_id in image_ids:
                download_queue.put(obj)

    # wait for the queue to be empty, then join all threads
    download_queue.join()
    for _ in range(num_threads):
        download_queue.put(None)
    for worker in workers:
        worker.join()

    # create a directory to store downloaded metadata
    labels_dir = data_dir / 'labels'
    try:
        os.mkdir(labels_dir)
    except FileExistsError:
        shutil.rmtree(labels_dir)
        os.mkdir(labels_dir)
    os.chdir(labels_dir)

    # create a queue for objects that need to be downloaded
    # and spawn threads to download them concurrently
    download_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=download_object, args=(download_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    # loop through the files in the bucket and filter for JSON metadata
    # files for only labeled images; add them to the queue
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(os.environ["LABELED_BUCKET_NAME"])
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("_labels.csv"):
            image_id = obj.key.replace(prefix, '').replace("_labels.csv", '')
            if image_id in image_ids:
                download_queue.put(obj)

    # wait for the queue to be empty, then join all threads
    download_queue.join()
    for _ in range(num_threads):
        download_queue.put(None)
    for worker in workers:
        worker.join()

    # create a directory to store downloaded metadata
    images_dir = data_dir / 'images'
    try:
        os.mkdir(images_dir)
    except FileExistsError:
        shutil.rmtree(images_dir)
        os.mkdir(images_dir)
    os.chdir(images_dir)

    # create a queue for objects that need to be downloaded
    # and spawn threads to download them concurrently
    download_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=download_object, args=(download_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    # loop through the files in the bucket and filter for JSON metadata
    # files for only labeled images; add them to the queue
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(os.environ["LABELED_BUCKET_NAME"])
    for obj in bucket.objects.filter(Prefix=prefix):
        file_extensions = [".png", ".jpg", ".jpeg"]
        for extension in file_extensions:
            if obj.key.endswith(extension):
                image_id = obj.key.replace(prefix, '').replace(extension, '')
                if image_id in image_ids:
                    download_queue.put(obj)

    # wait for the queue to be empty, then join all threads
    download_queue.join()
    for _ in range(num_threads):
        download_queue.put(None)
    for worker in workers:
        worker.join()

    os.chdir(cwd)


def get_s3_filepath(image_id, prefix="", filetype="png"):
    """Constructs the S3 filepath for a given image ID
    
    Args:
        image_id (str): the unique ID for the image and its associated labeled
            data
        prefix (str, optional): Defaults to "labeled". The S3 prefix to the
            filename
        filetype (str, optional): Defaults to "png". The filetype (suffix)
    
    Returns:
        str: the S3 url of the image
    """
    bucket_name = os.environ["LABELED_BUCKET_NAME"]
    key_name = "{prefix}/{id}.{suffix}".format(
        prefix=prefix, id=image_id, suffix=filetype)
    url = "https://s3.amazonaws.com/{bucket}/{key}".format(
        bucket=bucket_name, key=key_name)
    return url


def load_BBoxLabeledImages(image_ids, num_threads=20):
    """Loads a set of BBoxLabeledImage objects from the filesystem

    NOTE: this is done concurrently to limit I/O costs. Since no image data
    is stored in these objects (only dimensions and bounding-box details), it
    should not be too memory-intensive to store the entire dataset's
    bounding-box data in RAM.
    
    Args:
        image_ids (list): the list of image IDs that should be loaded
        num_threads (int, optional): Defaults to 20. The number of threads that
            should be used for concurrent loading.
    
    Returns:
        dict: a dict where the keys are image IDs and the values are the
            BBoxLabeledImage object for that image
    """
    bbox_labeled_images = {}

    # loads a BBoxLabeledImage object from the given image_id
    def load(image_id):
        labeled_image_mask = labeled_image_mask = LabeledImageMask.from_files(
            image_id)
        return BBoxLabeledImage.from_labeled_image_mask(labeled_image_mask)

    # pulls image_ids from a queue, loads the relevant object
    # NOTE: this is the function being performed concurrently
    def worker_load_func(queue):
        while True:
            image_id = queue.get()
            if image_id is None:
                break
            bbox_labeled_image = load(image_id)
            bbox_labeled_images[image_id] = bbox_labeled_image
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

    return bbox_labeled_images


def load_LabeledImageMasks(image_ids, num_threads=20):
    """Loads a set of LabeledImageMask objects from the filesystem

    NOTE: this is done concurrently to limit I/O costs.
    
    Args:
        image_ids (list): the list of image IDs that should be loaded
        num_threads (int, optional): Defaults to 20. The number of threads that
            should be used for concurrent loading.
    
    Returns:
        dict: a dict where the keys are image IDs and the values are the
            LabeledImageMask object for that image
    """
    labeled_image_masks = []

    # loads a LabeledImageMask object from the given image_id
    def load(image_id):
        labeled_image_mask = LabeledImageMask.from_files(image_id)
        return labeled_image_mask

    # pulls image_ids from a queue, loads the relevant object
    # NOTE: this is the function being performed concurrently
    def worker_load_func(queue):
        while True:
            image_id = queue.get()
            if image_id is None:
                break
            labeled_image_mask = load(image_id)
            labeled_image_masks.append(labeled_image_mask)
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

    return labeled_image_masks


def upload_dataset(bucket_name, directory, num_threads=20):
    """Recursively uploads a directory to S3
    
    Args:
        bucket_name (str): the name of the S3 bucket to upload to
        directory (str): the absolute path of a directory whose contents
            should be uploaded to S3; the directory name is used as the S3
            prefix for all uploaded files
        num_threads (int, optional): Defaults to 20.
    """
    s3 = boto3.resource('s3')

    def upload_file(queue):
        while True:
            obj = queue.get()
            if obj is None:
                break
            abspath, s3_path = obj
            s3.meta.client.upload_file(abspath, bucket_name, s3_path)
            queue.task_done()

    # create a queue for objects that need to be uploaded
    # and spawn threads to upload them concurrently
    upload_queue = Queue(maxsize=0)
    workers = []
    for worker in range(num_threads):
        worker = Thread(target=upload_file, args=(upload_queue, ))
        worker.setDaemon(True)
        worker.start()
        workers.append(worker)

    for root, _, files in os.walk(directory):
        for file in files:
            abspath = os.path.join(root, file)
            relpath = os.path.relpath(abspath, directory)
            s3_path = os.path.basename(directory) + "/" + relpath
            upload_queue.put((abspath, s3_path))

    # wait for the queue to be empty, then join all threads
    upload_queue.join()
    for _ in range(num_threads):
        upload_queue.put(None)
    for worker in workers:
        worker.join()


def get_bucket_folders(bucket_name, prefix):
    client = boto3.client('s3')
    result = client.list_objects(
        Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    folders = []
    for o in result.get('CommonPrefixes', []):
        folders.append(o.get('Prefix'))

    return folders
