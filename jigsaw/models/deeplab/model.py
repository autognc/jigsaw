import tensorflow as tf
import numpy as np
import cv2
import io
import PIL
from jigsaw.models.mask.model import LabeledImageMask

from deeplab.datasets import build_data


class DeeplabLabeledImage(LabeledImageMask):
    training_type = "Deeplab Semantic Segmentation"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def export_as_TFExample(self):
        """Converts object to tf_example

        Returns:
            tf_example (tf.train.Example): TensorFlow specified training object.
        """
        colors = np.array(list(self.label_masks.values()))  # shape (num_classes, 3)
        # get 1-indexed class IDs
        class_ids = np.array([DeeplabLabeledImage._label_to_int_dict[label] for label in self.label_masks.keys()])

        # read mask from disk
        mask = cv2.imread(self.mask_path)

        # convert mask to grayscale using numpy broadcasting magic
        binary_masks = (mask[..., None] == colors.T).all(axis=2)
        grayscale_mask = np.where(binary_masks, class_ids, 0).sum(axis=2)

        # convert to bytestring
        img = PIL.Image.fromarray(grayscale_mask.astype(np.uint8))
        mask_bytes = io.BytesIO()
        img.save(mask_bytes, format="PNG")

        # read real image from disk
        with tf.io.gfile.GFile(str(self.image_path), 'rb') as fid:
            real_bytes = fid.read()

        return build_data.image_seg_to_tfexample(
            real_bytes, str(self.image_path), self.ydim, self.xdim, mask_bytes.getvalue())
