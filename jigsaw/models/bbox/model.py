import numpy as np
import pandas as pd
import cv2

from pathlib import Path
import tensorflow as tf
import PIL
import io
import os
import xml.etree.ElementTree as ET

from colorama import init, Fore
from halo import Halo
from object_detection.utils import dataset_util

from jigsaw.data_interface import LabeledImage
from jigsaw.cli_utils import (user_confirms, user_input, user_selection,
                              FilenameValidator, IntegerValidator, Spinner)
from jigsaw.io_utils import copy_data_locally, download_data_from_s3
from jigsaw.models.bbox.filtering import (ingest_metadata, load_metadata, and_filter, or_filter, join_sets)
from jigsaw.models.bbox.transforms import (load_labels, perform_transforms, Transform)


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
    label_to_int_dict = {}

    associated_files = {
        "image": ".png",
        "metadata": "_meta.json",
        "mask": "_mask.png",
        "labels": "_labels.csv",
        "PASCAL_VOC_labels": "_labels.xml"
    }

    training_type = "Bounding Box"
    
    def __init__(self, image_id, image_path, label_boxes, xdim, ydim):
        self.image_id = image_id
        self.image_path = image_path
        self.label_boxes = label_boxes
        self.xdim = xdim
        self.ydim = ydim
    
    @classmethod
    def renumber_label_to_int_dict(cls):
        for i, label in enumerate(BBoxLabeledImage.label_to_int_dict.keys()):
            BBoxLabeledImage.label_to_int_dict[label] = i + 1
    
    @classmethod
    def delete_label_int(cls, label):
        if label in BBoxLabeledImage.label_to_int_dict.keys():
            del BBoxLabeledImage.label_to_int_dict[label]
            # renumber all values
            cls.renumber_label_to_int_dict()
    
    @classmethod
    def add_label_int(cls, label_to_add):
        if label_to_add not in BBoxLabeledImage.label_to_int_dict.keys():
            # add the new label
            BBoxLabeledImage.label_to_int_dict[label_to_add] = None

            # renumber all values
            cls.renumber_label_to_int_dict()

    @classmethod
    def from_PASCAL_VOC(cls, image_id, image_filepath, labels_filepath):
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
            box = BoundingBox(label, xmin, xmax, ymin, ymax)
            label_boxes.append(box)
        return BBoxLabeledImage(image_id, image_filepath, label_boxes, xdim, ydim)


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

    @classmethod
    def from_semantic_labels(cls, image_id, image_filepath, mask_filepath, labels_filepath, skip_background):
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
                    instance["xmin"], instance["xmax"], instance["ymin"], instance["ymax"])
                label_boxes.append(box)

        bbox = BBoxLabeledImage(image_id, image_filepath, label_boxes, xdim, ydim)
        os.remove(mask_filepath)
        os.remove(labels_filepath)
        bbox.save_changes()
        return bbox

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
        image_filepath = data_dir / str(image_id + ".png")

        labels_xml_path = data_dir / (str(image_id) + "_labels.xml")
        if labels_xml_path.exists():
            return cls.from_PASCAL_VOC(image_id, image_filepath, labels_xml_path)
        else:
            mask_filepath = data_dir / str(image_id + "_mask.png")
            labels_filepath = data_dir / str(image_id + "_labels.csv")
            
            return cls.from_semantic_labels(image_id, image_filepath, mask_filepath, labels_filepath, skip_background)

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
        self.label_boxes.append(BoundingBox(new_label, new_xmin, new_xmax, new_ymin, new_ymax))

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
        filename.text = self.image_id + ".png"
        path.text = self.image_id + ".png"
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
        
        xml_file = open(data_dir / (self.image_id + "_labels.xml"), 'w')
        xml_file.write(xml_data)
        xml_file.close()

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

    @classmethod
    def filter_and_load(cls, data_source, **kwargs):
        ingest_metadata(data_source, **kwargs)

        tags_df = load_metadata()
        filter_metadata = {"groups": []}
        
        # ask the user if they would like to perform filtering
        # if yes, enter a loop that supplies filter options
        # if no, skip
        if user_confirms(
                message="Would you like to filter out any of the data?",
                default=False):
            sets = {}
            # outer loop to determine how many sets the user will create
            while True:

                try:
                    subset = tags_df
                    this_group_filters = []
                    len_subsets = [len(subset)]
                    # inner loop to handle filtering for ONE set
                    while True:

                        # if filters have been applied, display them to the user
                        # to help guide their next choice
                        if len(this_group_filters) > 0:
                            filters_applied = [
                                "   > " + (" " + f["type"] + " ").join(f["tags"]) +
                                " ({} -> {})".format(len_subsets[i],
                                                    len_subsets[i + 1])
                                for i, f in enumerate(this_group_filters)
                            ]
                            print(Fore.MAGENTA + "ℹ Filters already applied:\n{}".
                                format("\n".join(filters_applied)))

                        selected_tags = user_selection(
                            message=
                            "Please select a set of tags with which to apply a filter:",
                            choices=list(tags_df),
                            selection_type="checkbox")
                        filter_type = user_selection(
                            message=
                            "Which filter would you like to apply to the above set?",
                            choices=["AND (intersection)", "OR (union)"],
                            selection_type="list")

                        if filter_type == "AND (intersection)":
                            subset = and_filter(subset, selected_tags)
                            this_group_filters.append({
                                "type": "AND",
                                "tags": selected_tags
                            })
                        elif filter_type == "OR (union)":
                            subset = or_filter(subset, selected_tags)
                            this_group_filters.append({
                                "type": "OR",
                                "tags": selected_tags
                            })
                        print(
                            Fore.GREEN +
                            "ℹ There are {} images that meet the filter criteria selected."
                            .format(len(subset)))
                        len_subsets.append(len(subset))

                        if not user_confirms(
                                message=
                                "Would you like to continue filtering this set?",
                                default=False):
                            set_name = user_input(
                                message="What would you like to name this set?",
                                validator=FilenameValidator)
                            sets[set_name] = subset
                            filter_metadata["groups"].append({
                                "name":
                                set_name,
                                "filters":
                                this_group_filters
                            })
                            break
                except:
                    print("Sorry, there were no tags on the data to filter by. Using all images.")
                    break

                if not user_confirms(
                        message=
                        "Would you like to create more sets via filtering?",
                        default=False):
                    break

            sets_to_join = []
            for set_name, set_data in sets.items():
                how_many = user_input(
                    message=
                    'How many images of set "{}" would you like to use? (?/{})'
                    .format(set_name, len(set_data)),
                    validator=IntegerValidator,
                    default=str(len(set_data)))
                n = int(how_many)
                sets_to_join.append(
                    set_data.sample(n, replace=False, random_state=42))

                # find the right group within the metadata dict and add the number
                # included to it
                for group in filter_metadata["groups"]:
                    if group["name"] == set_name:
                        group["number_included"] = n

            image_ids = join_sets(sets_to_join).index.tolist()

        else:
            image_ids = tags_df.index.tolist()

        def need_file(filename):
            for suffix in cls.associated_files.values():
                if not filename.endswith(suffix):
                    continue
                image_id = filename.rstrip(suffix)
                if image_id in image_ids:
                    return True

        if data_source == "Local":
            spinner = Spinner(
                text="Copying data locally into Jigsaw...",
                text_color="magenta")
            spinner.start()
            copy_data_locally(
                source_dir=kwargs["data_filepath"], condition_func=need_file)

        elif data_source == "S3":
            spinner = Spinner(
                text="Downloading data from S3...",
                text_color="magenta")
            spinner.start()
            download_data_from_s3(
                bucket_name=kwargs["bucket"], condition_func=need_file)

        spinner.succeed(text=spinner.text + "Complete.")

        return image_ids, filter_metadata

    @classmethod
    def transform(cls, image_ids, **kwargs):
        transform_list = []

        # ask the user if they would like to perform transforms
        # if yes, enter a loop that supplies transform options
        # if no, skip
        if user_confirms(
                message="Would you like to perform any data transformations?",
                default=False):
            spinner = Spinner(
                text="Loading image labels...", text_color="magenta")
            spinner.start()
            labels = load_labels()
            spinner.succeed(text=spinner.text + "Complete.")

            while True:
                transform = user_selection(
                    message="Which type of transformation?",
                    choices=["Label Renaming", "Label Merging", "Exit"],
                    sort_choices=False,
                    selection_type="list")

                if transform == "Exit":
                    break

                if transform == "Label Renaming":
                    while True:
                        original = user_selection(
                            message="Which label would you like to rename?",
                            choices=labels,
                            selection_type="list")
                        new = user_input(
                            message="What should the new name be?",
                            validator=FilenameValidator)

                        labels = [
                            new if label == original else label
                            for label in labels
                        ]
                        transform_list.append(
                            Transform(
                                transform_type="rename",
                                original=original,
                                new=new))

                        if not user_confirms(
                                message="Would you like to continue renaming?",
                                default=False):
                            break

                if transform == "Label Merging":
                    while True:
                        # insert transforms here
                        originals = user_selection(
                            message="Which labels should be merged into one?",
                            choices=labels,
                            selection_type="checkbox")
                        new = user_input(
                            message="What should the merged name be?",
                            validator=FilenameValidator)

                        for original in originals:
                            labels.remove(original)
                        labels.append(new)
                        transform_list.append(
                            Transform(
                                transform_type="merge",
                                original=originals,
                                new=new))

                        if not user_confirms(
                                message="Would you like to continue merging?",
                                default=False):
                            break

            spinner = Spinner(
                text="Performing transformations...", text_color="magenta")
            spinner.start()
            perform_transforms(transform_list, image_ids=image_ids)
            spinner.succeed(text=spinner.text + "Complete.")

        # collect metadata on transforms
        transform_metadata = []
        for transform in transform_list:
            transform_metadata.append({
                "type": transform.transform_type,
                "original": transform.original,
                "new": transform.new
            })

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
        for label_name, label_int in cls.label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))


class BoundingBox:
    """Stores the label and bounding box dimensions for a detected image region
    
    Attributes:
        label (str): the classification label for the region (e.g., "cygnus")
        xmin (int): the pixel location of the left edge of the bounding box
        xmax (int): the pixel location of the right edge of the bounding box
        ymin (int): the pixel location of the top edge of the bounding box
        ymax (int): the pixel location of the top edge of the bounding box
    """

    def __init__(self, label, xmin, xmax, ymin, ymax):
        self.label = label
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return "label: {} | xmin: {} | xmax: {} | ymin: {} | ymax: {}".format(
            self.label, self.xmin, self.xmax, self.ymin, self.ymax)
    
    @property
    def label_int(self):
        return BBoxLabeledImage.label_to_int_dict[self.label]