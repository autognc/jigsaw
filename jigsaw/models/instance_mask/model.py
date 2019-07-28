import PIL
import io
import os
import itertools
import cv2
import uuid
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from object_detection.utils import dataset_util
from jigsaw.model_utils.base.mask import LabeledImageMask
from jigsaw.model_utils.types import BoundingBox

class InstanceImageMask(LabeledImageMask):
    training_type = "Instance Segmentation"
    
    verbose_write = False
    
    def __init__(self, image_id, image_path, image_type, label_boxes, mask_path, label_masks, binary_masks,
                 xdim, ydim):
        super().__init__(image_id, image_path, image_type, mask_path, label_masks, xdim, ydim)
        self.label_boxes = label_boxes
        self.binary_masks = binary_masks
    
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
        
        label_boxes = []
        binary_masks = []
        image_mask = cv2.imread(mask_filepath)
        r = 2
        b = 0
        g = 1
        for label, color in label_masks.items():
            matched = False
            x = [-2 -1, 0, 1, 2]
            iters = [p for p in itertools.product(x, repeat=3)]
            mask = np.zeros(image_mask.shape, dtype=np.uint8)
            for i in iters:
                c = np.add(color, np.array(i))
                match = np.where((image_mask == c).all(axis=2))
                y, x = match
                if len(y) != 0 and len(x) != 0:

                    mask[match] = [255, 255, 255]
                    matched = True
                    
            cls.add_label_int(label)

            if not matched:
                continue
            
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            box = BoundingBox(label, xmin, xmax, ymin, ymax, cls)
            label_boxes.append(box)
            binary_masks.append(mask)

        return cls(image_id, image_filepath, image_type, label_boxes, mask_filepath,
                                label_masks, binary_masks, xdim, ydim)

    
    def export_as_TFExample(self):
        """Converts LabeledImageMask object to tf_example
        
        Returns:
            tf_example (tf.train.Example): TensorFlow specified training object.
        """
        path_to_image = Path(self.image_path)

        with tf.gfile.GFile(str(path_to_image.absolute()), 'rb') as fid:
            encoded_png = fid.read()

        image_width  = self.xdim
        image_height = self.ydim

        filename = path_to_image.name.encode('utf8')
        image_format = bytes(self.image_type, encoding='utf-8')
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

        feature_dict = {
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
        }

        
        encoded_list = []
        
        for binary_mask in self.binary_masks:
            img = PIL.Image.fromarray(binary_mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_list.append(output.getvalue())
            if self.verbose_write:
                os.makedirs(os.path.expanduser('~/Desktop/jigsaw_instance_masks/' + self.image_id), exist_ok=True)
                img.save(os.path.expanduser(f'~/Desktop/jigsaw_instance_masks/{self.image_id}/' + str(uuid.uuid4()) + '.png'), format='PNG')

        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_list))

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        
        return tf_example
            