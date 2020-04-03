import PIL
import io
import os
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from object_detection.utils import dataset_util
from jigsaw.data_interface import LabeledImage
from jigsaw.model_utils.filters import default_filter_and_load
from jigsaw.model_utils.transforms import default_perform_transforms
from jigsaw.constants import METADATA_PREFIX

class LabeledImageMask(LabeledImage):
    """Stores pixel-wise-labeled image data and provides related operations

    Attributes:
        image_id (str): the unique ID for the image and labeled data
        image_path (str): the path to the source image
        image_type(str): type of image (file extension)
        mask_path (str): the path to the semantic image mask
        label_masks (dict): a dict storing the labels (str) as keys and
            matching pixel colors (3x1 numpy array) in the image mask as values
        xdim (int): width of the image (in pixels)
        ydim (int): height of the image (in pixels)
    """
    _label_to_int_dict = {}

    associated_files = {
        "image_type_1": ".png",
        "image_type_2": ".jpg",
        "image_type_3": ".jpeg",
        "metadata": ".json",
        "labels": ".csv",
    }
    
    related_data_prefixes = {
        "meta": METADATA_PREFIX,
        'images': 'image_',
        'labels': 'labels_',
        'masks': 'mask_'
    }
    
    temp_dir = None

    def __init__(self, image_id, image_path, image_type, mask_path, label_masks, xdim,
                 ydim):
        super().__init__(image_id)
        self.image_path = image_path
        self.image_type = image_type
        self.mask_path = mask_path
        self.label_masks = label_masks
        for label in label_masks:
            self.add_label_int(label)
        self.xdim = xdim
        self.ydim = ydim
    
    
    ## CLASS METHODS ##
    @classmethod
    def construct(cls, image_id, **kwargs):
        """Constructs a LabeledImageMask object from a set of standard files
        
        Args:
            image_id (str): the unique ID for this image
        
        Returns:
            LabeledImageMask: the object representative of this semantically-
                labeled image
        """
        try:
            skip_background = kwargs["skip_background"]
        except KeyError:
            skip_background = True

        if cls.temp_dir is None:
            cwd = Path.cwd()
            data_dir = cwd / 'data'
        else:
            data_dir = cls.temp_dir
        
        mask_filepath = data_dir / f'mask_{image_id}.png'
        mask_filepath = str(
            mask_filepath.absolute())  # cv2.imread doesn't like Path objects.
        labels_filepath = data_dir / f'labels_{image_id}.csv'
        
        image_filepath = None
        image_type = None
        file_extensions = [".png", ".jpg", ".jpeg"]
        for extension in file_extensions:
            temp_filepath = data_dir / f'image_{image_id}{extension}'
            if os.path.exists(data_dir / f'image_{image_id}{extension}'):
                image_filepath = data_dir / f'image_{image_id}{extension}'
                image_type = extension
                break

        if image_filepath is None:
            raise ValueError("Hmm, there doesn't seem to be a valid image filepath.")

        labels_df = pd.read_csv(labels_filepath, index_col="label")
        image_mask = cv2.imread(mask_filepath)
        ydim, xdim, _ = image_mask.shape

        label_masks = {}
        for label, color in labels_df.iterrows():
            if label == "Background" and skip_background:
                continue
            color_bgr = np.array([color["B"], color["G"], color["R"]])
            label_masks[label] = color_bgr
            
        return cls(image_id, image_filepath, image_type, mask_filepath,
                                label_masks, xdim, ydim)
    
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
        cls.write_label_map(dataset_name)

    @classmethod
    def write_label_map(cls, dataset_name):
        """Writes out the TensorFlow Object Detection Label Map
        
        Args:
            dataset_name (str): the name of the dataset
        """
        dataset_path = Path.cwd() / 'dataset' / dataset_name
        label_map_filepath = dataset_path / 'label_map.pbtxt'
        label_map = []
        for label_name, label_int in cls._label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))
            
    
    ## TRANSFORMATIONS ##
    def rename_label(self, original_label, new_label):
        """Renames a given label
        
        Args:
            original_label (str): the original label that should be renamed
            new_label (str): the new label name
        """
        # if the new label already exists, treat this as a merge
        if new_label in self.label_masks.keys():
            self.merge_labels([original_label, new_label], new_label)
            return

        # perform rename within the label_masks dict object attribute
        try:
            label_color = self.label_masks[original_label]
            del self.label_masks[original_label]
            self.label_masks[new_label] = label_color
            self.save_label_changes()
        except KeyError:
            # this image does not have an instance of the original_label
            return

        # handle the class attribute for label_to_int conversions
        self.delete_label_int(original_label)
        self.add_label_int(new_label)

    def merge_labels(self, original_labels, new_label):
        """Merges a set of existing labels into one label

        Args:
            original_labels (list): a list of the labels (str) that should be
                merged into a single label
            new_label (str): the merged label name
        """
        # if the new_label already exists for any reason, add it to the list
        # of labels to be merged to avoid any issues
        if new_label in self.label_masks.keys():
            original_labels.append(new_label)

        merged_color = np.random.randint(256, size=3)
        while True:
            color_already_exists = False
            for color in self.label_masks.values():
                if np.array_equal(merged_color, color):
                    color_already_exists = True
            if color_already_exists:
                merged_color = np.random.randint(256, size=3)
            else:
                break

        image_mask = cv2.imread(self.mask_path)

        for label_to_merge in original_labels:
            try:
                label_color = self.label_masks[label_to_merge]
                match = np.where((image_mask == label_color).all(axis=2))
                image_mask[match] = merged_color
                del self.label_masks[label_to_merge]
            except KeyError:
                pass
            self.delete_label_int(label_to_merge)

        self.add_label_int(new_label)

        self.label_masks[new_label] = merged_color
        self.save_mask_changes(image_mask)
        self.save_label_changes()

    def save_label_changes(self):
        """Saves label changes to corresponding self LabeledImageMask

        """
        if self.temp_dir is None:
            cwd = Path.cwd()
            data_dir = cwd / 'data'
        else:
            data_dir = self.temp_dir

        labels_filepath = data_dir / f'labels_{self.image_id}.csv'
        
        lines = ["label,R,G,B"]
        for label, color in self.label_masks.items():
            lines.append("{lab},{R},{G},{B}".format(
                lab=label, R=color[2], G=color[1], B=color[0]))
        to_write = "\n".join(lines)
        with open(labels_filepath, "w") as labels_csv:
            labels_csv.write(to_write)

    def save_mask_changes(self, changed_mask):
        """Overwrites existing mask for corresponding LabeledImageMask with changed_mask

        Args:
            changed_mask (cv2 image representation): modified mask to write out
        """
        if self.temp_dir is None:
            cwd = Path.cwd()
            data_dir = cwd / 'data'
        else:
            data_dir = self.temp_dir
            
        mask_filepath = data_dir / f'mask_{self.image_id}.png'

        cv2.imwrite(str(mask_filepath.absolute()), changed_mask)

    
    ## EXPORTER ##
    def export_as_TFExample(self):
        """Converts LabeledImageMask object to tf_example
        
        Returns:
            tf_example (tf.train.Example): TensorFlow specified training object.
        """
        path_to_image = Path(self.image_path)

        with tf.gfile.GFile(str(path_to_image.absolute()), 'rb') as fid:
            encoded_png = fid.read()

        image_width = self.xdim
        image_height = self.ydim

        filename = path_to_image.name.encode('utf8')
        image_format = bytes(self.image_type, encoding='utf-8')

        masks = []
        classes_text = []
        classes = []

        image_mask = cv2.imread(self.mask_path)
        for class_label, color in self.label_masks.items():
            matches = np.where(
                (image_mask == color).all(axis=2), 1, 0.0).astype(np.uint8)
            classes_text.append(class_label.encode('utf8'))
            classes.append(self._label_to_int_dict[class_label])
            masks.append(matches)

        encoded_mask_png_list = []

        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height':
                    dataset_util.int64_feature(image_height),
                    'image/width':
                    dataset_util.int64_feature(image_width),
                    'image/filename':
                    dataset_util.bytes_feature(filename),
                    'image/source_id':
                    dataset_util.bytes_feature(filename),
                    'image/encoded':
                    dataset_util.bytes_feature(encoded_png),
                    'image/format':
                    dataset_util.bytes_feature(image_format),
                    'image/object/class/text':
                    dataset_util.bytes_list_feature(classes_text),
                    'image/object/class/label':
                    dataset_util.int64_list_feature(classes),
                    'image/object/mask':
                    dataset_util.bytes_list_feature(encoded_mask_png_list),
                }))

        return tf_example

    
    ## HELPERS ##
    @classmethod
    def renumber_label_to_int_dict(cls):
        for i, label in enumerate(LabeledImageMask._label_to_int_dict.keys()):
            LabeledImageMask._label_to_int_dict[label] = i + 1

    @classmethod
    def delete_label_int(cls, label):
        if label in LabeledImageMask._label_to_int_dict.keys():
            del LabeledImageMask._label_to_int_dict[label]
            # renumber all values
            cls.renumber_label_to_int_dict()

    @classmethod
    def add_label_int(cls, label_to_add):
        if label_to_add not in LabeledImageMask._label_to_int_dict.keys():
            # add the new label
            LabeledImageMask._label_to_int_dict[label_to_add] = None
            # renumber all values
            cls.renumber_label_to_int_dict()
