import tensorflow as tf
import numpy as np
import os
import cv2
import json
from pathlib import Path

from object_detection.utils import dataset_util
from jigsaw.data_interface import LabeledImage
from jigsaw.model_utils.filters import default_filter_and_load
from jigsaw.model_utils.transforms import default_perform_transforms
from jigsaw.model_utils.types import BoundingBox
from jigsaw.constants import METADATA_PREFIX


class PoseRegression(LabeledImage):
    training_type = "Pose Regression"

    associated_files = {
        "image_type_1": ".png",
        "image_type_2": ".jpg",
        "image_type_3": ".jpeg",
        "metadata": ".json"
    }

    related_data_prefixes = {
        "meta": METADATA_PREFIX,
        'images': 'image_',
    }

    temp_dir = None
    _feature_point_labels = None
    _label_to_int_dict = {}
    _aggregate = None

    def __init__(self, image_id, image_path, image_type, label_boxes, pose, xdim, ydim):
        super().__init__(image_id)
        self.image_path = image_path
        self.image_type = image_type
        self.pose = pose
        self.label_boxes = label_boxes
        self.xdim = xdim
        self.ydim = ydim

    @classmethod
    def construct(cls, image_id, **kwargs):
        if cls.temp_dir is None:
            cwd = Path.cwd()
            data_dir = cwd / 'data'
        else:
            data_dir = cls.temp_dir

        image_filepath = None
        image_type = None
        image_extensions = [v for k, v in cls.associated_files.items() if 'image' in k]
        for extension in image_extensions:
            image_filepath = data_dir / f'{cls.related_data_prefixes["images"]}{image_id}{extension}'
            if os.path.exists(image_filepath):
                image_type = extension[1:]
                break

        if image_filepath is None:
            raise ValueError("Hmm, there doesn't seem to be a valid image filepath.")

        meta_filepath = data_dir / f'{cls.related_data_prefixes["meta"]}{image_id}{cls.associated_files["metadata"]}'
        if not os.path.exists(meta_filepath):
            raise ValueError("Hmm, there doesn't seem to be a valid metadata filepath.")

        with open(data_dir / f"meta_{image_id}.json", "r") as f:
            meta = json.load(f)

        # if this is the first image, load the feature point labels
        # if not cls._feature_point_labels:
            # feature_points = meta['truth_centroids']
            # cls._feature_point_labels = sorted(feature_points.keys())

        label_boxes = []
        for label, bbox in meta['bboxes'].items():
            label_boxes.append((label, bbox))
            if label not in cls._label_to_int_dict.keys():
                cls._label_to_int_dict[label] = max(cls._label_to_int_dict.values()) + 1 if cls._label_to_int_dict else 0

        image = cv2.imread(str(image_filepath.absolute()))
        ydim, xdim, channels = image.shape

        if not cls._aggregate:
            mean = np.zeros([channels], dtype=np.float32)
            m2 = np.zeros([channels], dtype=np.float32)
            cls._aggregate = (0, mean, m2)

        # aggregate the mean and squared distance from mean using
        # Welford's algorithm (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
        count, mean, m2 = cls._aggregate
        count += 1
        reduced_image = np.mean(image, axis=(0, 1))
        delta = reduced_image - mean
        mean += delta / count
        delta2 = reduced_image - mean
        m2 += delta * delta2
        cls._aggregate = (count, mean, m2)
        return cls(image_id, image_filepath, image_type, label_boxes, meta['pose'], xdim, ydim)

    @classmethod
    def filter_and_load(cls, data_source, **kwargs):
        image_ids, filter_metadata, temp_dir = default_filter_and_load(data_source=data_source, **kwargs)
        return image_ids, filter_metadata, temp_dir

    @classmethod
    def transform(cls, image_ids, **kwargs):
        transform_metadata = default_perform_transforms(image_ids, cls, cls.temp_dir, **kwargs)
        return transform_metadata

    @classmethod
    def write_additional_files(cls, dataset_name, **kwargs):
        # output_path = Path.cwd() / 'dataset' / dataset_name / 'feature_points.json'
        # with open(output_path, 'w') as f:
            # json.dump(cls._feature_point_labels, f)

        mean_path = Path.cwd() / 'dataset' / dataset_name / 'mean.npy'
        stdev_path = Path.cwd() / 'dataset' / dataset_name / 'stdev.npy'
        count, mean, m2 = cls._aggregate
        np.save(str(mean_path), mean)
        np.save(str(stdev_path), np.sqrt(m2 / count))

    def export_as_TFExample(self):
        with tf.gfile.GFile(str(self.image_path), 'rb') as fid:
            encoded_png = fid.read()

        # feature_points = metadata['truth_centroids']


        # if sorted(feature_points.keys()) != self._feature_point_labels:
            # raise ValueError(
                # f"File {self.image_path} contains inconsistent feature points: expected {self._feature_point_labels}, got {sorted(feature_points.keys())}"
            # )
        # sort by key to make sure always the same order
        # feature_points = [feature_points[k] for k in self._feature_point_labels]
        # feature_points = [x[0] for x in feature_points] + [x[1] for x in feature_points]

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height':
                        dataset_util.int64_feature(self.ydim),
                    'image/width':
                        dataset_util.int64_feature(self.xdim),
                    'image/source_id':
                        dataset_util.bytes_feature(self.image_id.encode('utf-8')),
                    'image/encoded':
                        dataset_util.bytes_feature(encoded_png),
                    'image/format':
                        dataset_util.bytes_feature(self.image_type.encode('utf-8')),
                    # 'feature_points':
                        # dataset_util.int64_list_feature(feature_points),
                    'image/object/bbox/xmin': dataset_util.float_list_feature([b['xmin'] / self.xdim for l, b in self.label_boxes]),
                    'image/object/bbox/xmax': dataset_util.float_list_feature([b['xmax'] / self.xdim for l, b in self.label_boxes]),
                    'image/object/bbox/ymin': dataset_util.float_list_feature([b['ymin'] / self.ydim for l, b in self.label_boxes]),
                    'image/object/bbox/ymax': dataset_util.float_list_feature([b['ymax'] / self.ydim for l, b in self.label_boxes]),
                    'image/object/class/text': dataset_util.bytes_list_feature([l.encode('utf-8') for l, b in self.label_boxes]),
                    'image/object/class/label': dataset_util.int64_list_feature([self._label_to_int_dict[l] for l, b in self.label_boxes]),
                    'image/object/pose': dataset_util.float_list_feature(self.pose)
                }))
