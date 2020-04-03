import io
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
from pathlib import Path
from object_detection.utils import dataset_util
from jigsaw.data_interface import LabeledImage
from jigsaw.model_utils.filters import default_filter_and_load
from jigsaw.model_utils.transforms import default_perform_transforms
from jigsaw.model_utils.types import BoundingBox
from jigsaw.constants import METADATA_PREFIX

class BBoxLabeledImage(LabeledImage):
    """Stores bounding-box-labeled image data and provides related operations

    Attributes:
        image_id (str): the unique ID for the image and labeled data
        image_path (str): the path to the source image
        label_boxes (list): a list of BoundingBox objects that store the labels
            and dimensions of each bounding box in the image
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
        "PASCAL_VOC_labels": ".xml"
    }
    
    related_data_prefixes = {
        "meta": METADATA_PREFIX,
        'images': 'image_',
        'labels': 'bboxLabels_'
    }

    temp_dir = None

    training_type = "Bounding Box"
    
    def __init__(self, image_id, image_path, image_type, label_boxes, xdim, ydim):
        super().__init__(image_id)
        self.image_id = image_id
        self.image_path = image_path
        self.image_type = image_type
        self.label_boxes = label_boxes
        self.xdim = xdim
        self.ydim = ydim
    
    
    ## CLASS METHODS ##
    @classmethod
    def construct(cls, image_id, **kwargs):
        """Constructs a BBoxLabeledImage object from a set of standard files
        
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

        cwd = Path.cwd()
        data_dir = cwd / 'data'

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

        labels_xml_path = data_dir / f'labels_{image_id}.xml'
        if labels_xml_path.exists():
            return cls.from_PASCAL_VOC(image_id, image_filepath, image_type, labels_xml_path)
        else:
            mask_filepath = data_dir / f'mask_{image_id}.png'
            labels_filepath = data_dir / f'labels_{image_id}.csv'
            
            return cls.from_semantic_labels(image_id, image_filepath, image_type, mask_filepath, labels_filepath, skip_background)
    
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

    @classmethod
    def from_semantic_labels(cls, image_id, image_filepath, image_type, mask_filepath, labels_filepath, skip_background):
        mask_filepath = str(
            mask_filepath.absolute())  # cv2.imread doesn't like Path objects.

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
        image_mask = cv2.imread(mask_filepath)
        for label, color in label_masks.items():
            match = np.where((image_mask == color).all(axis=2))
            y, x = match
            if len(y) == 0 or len(x) == 0:
                continue
            cls.add_label_int(label)

            mask = np.zeros(image_mask.shape, dtype=np.uint8)
            mask[match] = (255, 255, 255)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            instances = cls.segment_by_instance(mask)
            for instance in instances:
                box = BoundingBox(label,
                    instance["xmin"], instance["xmax"], instance["ymin"], instance["ymax"], cls)
                label_boxes.append(box)

        bbox = BBoxLabeledImage(image_id, image_filepath, image_type, label_boxes, xdim, ydim)
        bbox.save_changes()
        return bbox

    @classmethod
    def from_PASCAL_VOC(cls, image_id, image_filepath, image_type, labels_filepath):
        label_boxes = []
        tree = ET.parse(str(labels_filepath.absolute()))
        root = tree.getroot()
        size = root.find("size")
        xdim = int(size.find("width").text)
        ydim = int(size.find("height").text)
        for member in root.findall('object'):
            label = member.find("name").text
            bndbox = member.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find("ymax").text)
            cls.add_label_int(label)
            box = BoundingBox(label, xmin, xmax, ymin, ymax, cls)
            label_boxes.append(box)
        return BBoxLabeledImage(image_id, image_filepath, image_type, label_boxes, xdim, ydim)

    
    ## TRANSFORMATIONS ##
    def rename_label(self, original_label, new_label):
        """Renames a given label
        
        Args:
            original_label (str): the original label that should be renamed
            new_label (str): the new label name
        """
        # if the new label already exists, treat this as a merge
        label_already_exists = new_label in [box.label for box in self.label_boxes]
        if label_already_exists:
            self.merge_labels([original_label, new_label], new_label)
            return

        for box in self.label_boxes:
            if box.label == original_label:
                box.label = new_label
        self.save_changes()

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
        label_already_exists = new_label in [box.label for box in self.label_boxes]
        if label_already_exists:
            original_labels.append(new_label)
        
        new_label_boxes = []
        for i, box in enumerate(self.label_boxes):
            if box.label in original_labels:
                try:
                    new_xmin = min([box.xmin, new_xmin])
                    new_xmax = max([box.xmax, new_xmax])
                    new_ymin = min([box.ymin, new_ymin])
                    new_ymax = max([box.ymax, new_ymax])
                except NameError:
                    new_xmin = box.xmin
                    new_xmax = box.xmax
                    new_ymin = box.ymin
                    new_ymax = box.ymax
            else:
                new_label_boxes.append(box)
        self.label_boxes = new_label_boxes

        for label_to_merge in original_labels:
            self.delete_label_int(label_to_merge)

        self.add_label_int(new_label)
        self.label_boxes.append(BoundingBox(new_label, new_xmin, new_xmax, new_ymin, new_ymax, cls))

        self.save_changes()

    def save_changes(self):
        cwd = Path.cwd()
        data_dir = cwd / 'data'

        annotation = ET.Element('annotation')
        folder = ET.SubElement(annotation, 'folder')
        filename = ET.SubElement(annotation, 'filename')
        path = ET.SubElement(annotation, 'path')
        source = ET.SubElement(annotation, 'source')
        database = ET.SubElement(source, 'database')
        size = ET.SubElement(annotation, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        segmented = ET.SubElement(annotation, 'segmented')

        annotation.set('verified', 'yes')
        folder.text = 'images'
        filename.text = self.image_id + self.image_type
        path.text = self.image_id + self.image_type
        database.text = 'Unknown'
        width.text = str(self.xdim)
        height.text = str(self.ydim)
        depth.text = '3'
        segmented.text = '0'

        for box in self.label_boxes:
            object = ET.SubElement(annotation, 'object')
            name = ET.SubElement(object, 'name')
            pose = ET.SubElement(object, 'pose')
            truncated = ET.SubElement(object, 'truncated')
            difficult = ET.SubElement(object, 'difficult')
            bndbox = ET.SubElement(object, 'bndbox')
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            name.text = box.label
            pose.text = 'Unspecified'
            truncated.text = '0'
            difficult.text = '0'
            xmin.text = str(box.xmin)
            ymin.text = str(box.ymin)
            xmax.text = str(box.xmax)
            ymax.text = str(box.ymax)

        xml_data = str(ET.tostring(annotation, encoding='unicode'))
        
        xml_file = open(data_dir / f'bboxLabels_{self.image_id}.xml', 'w')
        xml_file.write(xml_data)
        xml_file.close()

    
    ## EXPORTER ##
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


    ## HELPERS ##
    @classmethod
    def renumber_label_to_int_dict(cls):
        for i, label in enumerate(BBoxLabeledImage._label_to_int_dict.keys()):
            BBoxLabeledImage._label_to_int_dict[label] = i + 1
    
    @classmethod
    def delete_label_int(cls, label):
        if label in BBoxLabeledImage._label_to_int_dict.keys():
            del BBoxLabeledImage._label_to_int_dict[label]
            # renumber all values
            cls.renumber_label_to_int_dict()
    
    @classmethod
    def add_label_int(cls, label_to_add):
        if label_to_add not in BBoxLabeledImage._label_to_int_dict.keys():
            # add the new label
            BBoxLabeledImage._label_to_int_dict[label_to_add] = None

            # renumber all values
            cls.renumber_label_to_int_dict()
    
    @classmethod
    def segment_by_instance(cls, mask):
        instances = []
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            instances.append({"xmin": x, "xmax": x + w, "ymin": y, "ymax": y + h})
        return instances
