import tensorflow as tf
import os
import cv2
import json
from pathlib import Path

from object_detection.utils import dataset_util
from jigsaw.data_interface import LabeledImage
from jigsaw.model_utils.filters import default_filter_and_load
from jigsaw.model_utils.transforms import default_perform_transforms
from jigsaw.constants import METADATA_PREFIX

class FeaturePointsRegression(LabeledImage):
    training_type = "Feature Points Regression"

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
    _num_feature_points = None

    def __init__(self, image_id, image_path, image_type, meta_path, xdim, ydim):
        super().__init__(image_id)
        self.image_path = image_path
        self.image_type = image_type
        self.xdim = xdim
        self.ydim = ydim
        self.meta_path = meta_path

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

        with open(meta_filepath, 'r') as f:
            feature_points = json.load(f)['truth_centroids']
        cls._num_feature_points = len(feature_points)

        ydim, xdim, _ = cv2.imread(str(image_filepath.absolute())).shape
        return cls(image_id, image_filepath, image_type, meta_filepath, xdim, ydim)

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
        output_path = Path.cwd() / 'dataset' / dataset_name / 'num_feature_points.txt'
        with open(output_path, 'w') as f:
            f.write(cls._num_feature_points)
            f.write('\n')

    def export_as_TFExample(self):
        with tf.gfile.GFile(str(self.image_path), 'rb') as fid:
            encoded_png = fid.read()

        with open(self.meta_path, 'r') as f:
            feature_points = json.load(f)['truth_centroids']
        if len(feature_points) != self._num_feature_points:
            raise ValueError(
                f"File {self.image_path} contains the wrong number of feature points: expected {self._num_feature_points}, got {len(feature_points)}"
            )
        # sort by key to make sure always the same order
        feature_points = sorted(feature_points.items())

        feature_point_labels = [f[0].encode('utf-8') for f in feature_points]
        feature_point_xcoords = [f[1][0] for f in feature_points]
        feature_point_ycoords = [f[1][1] for f in feature_points]

        return tf.train.Example(
            features=tf.train.Features(
                feature={
                    'height':
                        dataset_util.int64_feature(self.ydim),
                    'width':
                        dataset_util.int64_feature(self.xdim),
                    'image_id':
                        dataset_util.bytes_feature(self.image_id.encode('utf-8')),
                    'image_data':
                        dataset_util.bytes_feature(encoded_png),
                    'image_format':
                        dataset_util.bytes_feature(self.image_type.encode('utf-8')),
                    'feature_point_labels':
                        dataset_util.bytes_list_feature(feature_point_labels),
                    'feature_points':
                        dataset_util.int64_list_feature(feature_point_xcoords + feature_point_ycoords)
                }))
