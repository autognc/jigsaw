from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import cv2

from pathlib import Path
import tensorflow as tf
from object_detection.utils import dataset_util
import PIL
import io


class LabeledImage(ABC):

    associated_files = {"metadata": "_meta.json"}

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def export_as_TFExample(self, *args, **kwargs):
        pass
    


class Mask(LabeledImage):
    """Stores pixel-wise-labeled image data and provides related operations

    NOTE: Standard construction of a Mask object comes from the
    `from_files` method rather than the `__init__` method. This is because we
    can provide better functionality through multiple classmethod constructors
    (some of which may be added later.)

    Attributes:
        image_id (str): the unique ID for the image and labeled data
        image_path (str): the path to the source image
        mask_path (str): the path to the semantic image mask
        label_masks (dict): a dict storing the labels (str) as keys and
            matching pixel colors (3x1 numpy array) in the image mask as values
        xdim (int): width of the image (in pixels)
        ydim (int): height of the image (in pixels)
    """
    label_to_int_dict = {}

    def __init__(self, image_id, image_path, mask_path, label_masks, xdim,
                 ydim):
        self.image_id = image_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.label_masks = label_masks
        for label in label_masks:
            self.add_label_int(label)
        self.xdim = xdim
        self.ydim = ydim

    def renumber_label_to_int_dict(self):
        for i, label in enumerate(Mask.label_to_int_dict.keys()):
                Mask.label_to_int_dict[label] = i+1

    def delete_label_int(self, label):
        if label in Mask.label_to_int_dict.keys():
            del Mask.label_to_int_dict[label]
            # renumber all values
            self.renumber_label_to_int_dict()

    def add_label_int(self, label_to_add):
        if label_to_add not in Mask.label_to_int_dict.keys():
            # add the new label
            Mask.label_to_int_dict[label_to_add] = None
            
            # renumber all values
            self.renumber_label_to_int_dict()


    @classmethod
    def load(cls, image_id, skip_background=True):
        """Constructs a Mask object from a set of standard files
        
        Args:
            image_id (str): the unique ID for this image
            skip_background (bool, optional): Defaults to True. Whether to
                include "Background" as a formal class
        
        Returns:
            Mask: the object representative of this semantically-
                labeled image
        """
        cwd = Path.cwd()
        data_dir = cwd / 'data'
        masks_dir = data_dir / 'masks'
        labels_dir = data_dir / 'labels'
        images_dir = data_dir / 'images'
        mask_filepath = masks_dir / str(image_id + "_mask.png")
        mask_filepath = str(mask_filepath.absolute()) # cv2.imread doesn't like Path objects.
        labels_filepath = labels_dir / str(image_id + "_labels.csv")
        image_filepath = images_dir / str(image_id + ".jpg")

        labels_df = pd.read_csv(labels_filepath, index_col="label")
        image_mask = cv2.imread(mask_filepath)
        ydim, xdim, _ = image_mask.shape

        label_masks = {}
        for label, color in labels_df.iterrows():
            if label == "Background" and skip_background:
                continue
            color_bgr = np.array([color["B"], color["G"], color["R"]])
            label_masks[label] = color_bgr
        return Mask(image_id, image_filepath, mask_filepath,
                                label_masks, xdim, ydim)

    def rename_label(self, original_label, new_label):
        """Renames a given label
        
        Args:
            original_label (str): the original label that should be renamed
            new_label (str): the new label name
        """
        # if the new label already exists, treat this as a merge
        if new_label in self.label_masks.keys():
            self.merge_labels([original_label, new_label], new_label)

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
        """Saves label changes to corresponding self Mask

        """
        cwd = Path.cwd()
        data_dir = cwd / 'data'
        labels_dir = data_dir / 'labels'
        labels_filepath = labels_dir / str(self.image_id + "_labels.csv")

        lines = ["label,R,G,B"]
        for label, color in self.label_masks.items():
            lines.append("{lab},{R},{G},{B}".format(
                lab=label, R=color[2], G=color[1], B=color[0]))
        to_write = "\n".join(lines)
        with open(labels_filepath, "w") as labels_csv:
            labels_csv.write(to_write)

    def save_mask_changes(self, changed_mask):
        """Overwrites existing mask for corresponding Mask with changed_mask

        Args:
            changed_mask (cv2 image representation): modified mask to write out
        """
        cwd = Path.cwd()
        data_dir = cwd / 'data'
        masks_dir = data_dir / 'masks'
        mask_filepath = masks_dir / str(self.image_id + "_mask.png")

        cv2.imwrite(str(mask_filepath.absolute()), changed_mask)
            
    def export_as_TFExample(self):
        """Converts Mask object to tf_example
        
        Returns:
            tf_example (tf.train.Example): TensorFlow specified training object.
        """
        path_to_image = Path(self.image_path)

        with tf.gfile.GFile(str(path_to_image.absolute()), 'rb') as fid:
            encoded_jpg = fid.read()

        image_width  = self.xdim
        image_height = self.ydim

        filename = path_to_image.name.encode('utf8')
        image_format = b'jpg'

        masks = []
        classes_text = []
        classes = []

        image_mask = cv2.imread(self.mask_path)
        for class_label, color in self.label_masks.items():
            matches = np.where((image_mask == color).all(axis=2), 1, 0.0).astype(np.uint8)
            classes_text.append(class_label.encode('utf8'))
            classes.append(self.label_to_int_dict[class_label])
            masks.append(matches)

        encoded_mask_png_list = []

        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image_height),
            'image/width': dataset_util.int64_feature(image_width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/mask': dataset_util.bytes_list_feature(encoded_mask_png_list),
        }))

        return tf_example