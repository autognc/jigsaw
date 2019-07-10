import numpy as np

import cv2

from pathlib import Path
import tensorflow as tf
from object_detection.utils import dataset_util

from jigsaw.mask import LabeledImageMask

class BBoxLabeledImage:
    """Stores bounding-box-labeled image data and provides related operations

    NOTE: Standard construction of a BBoxLabeledImage object comes from the
    `from_labeled_image_mask` method rather than the `__init__` method. This 
    is because we can provide better functionality through multiple classmethod
    constructors (some of which may be added later.) The only currently
    recommended form of construction is directly from a LabeledImageMask.

    Attributes:
        image_id (str): the unique ID for the image and labeled data
        image_path (str): the path to the source image
        label_boxes (list): a list of BoundingBox objects that store the labels
            and dimensions of each bounding box in the image
        xdim (int): width of the image (in pixels)
        ydim (int): height of the image (in pixels)
    """

    def __init__(self, image_id, image_path, label_boxes, xdim, ydim):
        self.image_id = image_id
        self.image_path = image_path
        self.label_boxes = label_boxes
        self.xdim = xdim
        self.ydim = ydim

    @classmethod
    def from_labeled_image_mask(cls, labeled_image_mask):
        """Constructs a BBoxLabeledImage from a LabeledImageMask

        This converts semantically-labeled images to images labeled with
        bounding boxes for object detection.
        
        Args:
            labeled_image_mask (LabeledImageMask): an object representative of
                a semantically-labeled image
        
        Returns:
            BBoxLabeledImage: the equivalent image converted to have bounding
                box labels instead of full pixel-wise labels
        """
        label_boxes = []
        image_mask = cv2.imread(labeled_image_mask.mask_path)
        for label, color in labeled_image_mask.label_masks.items():
            match = np.where((image_mask == color).all(axis=2))
            y, x = match
            box = BoundingBox(label, LabeledImageMask.label_to_int_dict[label], x.min(), x.max(), y.min(), y.max())
            label_boxes.append(box)
        return BBoxLabeledImage(labeled_image_mask.image_id,
                                labeled_image_mask.image_path, label_boxes,
                                labeled_image_mask.xdim, labeled_image_mask.ydim)

    def convert_to_tf_example(self):
        """Converts BBoxLabeledImage object to tf_example
        
        Returns:
            tf_example (tf.train.Example): TensorFlow specified training object.
        """
        path_to_image = Path(self.image_path)

        with tf.gfile.GFile(str(path_to_image.absolute()), 'rb') as fid:
            encoded_png = fid.read()

        image_width  = self.xdim
        image_height = self.ydim

        filename = path_to_image.name.encode('utf8')
        image_format = b'png'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for bounding_box in self.label_boxes:
            xmins.append(bounding_box.xmin / image_width)
            xmaxs.append(bounding_box.xmax / image_width)
            ymins.append(bounding_box.ymin / image_height)
            ymaxs.append(bounding_box.ymax / image_height)
            classes_text.append(bounding_box.label.encode('utf8'))
            classes.append(bounding_box.label_int)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(image_height),
            'image/width': dataset_util.int64_feature(image_width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_png),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

        return tf_example

    def convert_to_dict(self):
        """Converts BBoxLabeledImage object to a dictionary representation of itself
        
        Returns:
            to_return (dict): dictionary representation of object
        """
        to_return = {}

        to_return['image_id'] = str(self.image_id)
        to_return['xdim'] = int(self.xdim)
        to_return['ydim'] = int(self.ydim)

        for index, boundingBox in enumerate(self.label_boxes):
            to_return[str('box_' + str(index))] = boundingBox.convert_to_dict()

        return to_return

class BoundingBox:
    """Stores the label and bounding box dimensions for a detected image region
    
    Attributes:
        label (str): the classification label for the region (e.g., "cygnus")
        xmin (int): the pixel location of the left edge of the bounding box
        xmax (int): the pixel location of the right edge of the bounding box
        ymin (int): the pixel location of the top edge of the bounding box
        ymax (int): the pixel location of the top edge of the bounding box
    """

    def __init__(self, label, label_int, xmin, xmax, ymin, ymax):
        self.label = label
        self.label_int = label_int
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return "label: {} | xmin: {} | xmax: {} | ymin: {} | ymax: {}".format(
            self.label, self.xmin, self.xmax, self.ymin, self.ymax)

    def convert_to_dict(self):
        """Converts BoundingBox object to a dictionary representation of itself
        
        Returns:
            to_return (dict): dictionary representation of object
        """
        to_return = vars(self)

        #Can't JSON out numpy custom types.
        for key, value in to_return.items():
            if isinstance(value, np.int64):
                to_return[key] = int(value)
        
        return to_return