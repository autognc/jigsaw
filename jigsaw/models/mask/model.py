import numpy as np
import pandas as pd
import cv2

from pathlib import Path
import tensorflow as tf
import PIL
import io

from colorama import init, Fore
from object_detection.utils import dataset_util

from jigsaw.data_interface import LabeledImage
from jigsaw.cli_utils import (user_confirms, user_input, user_selection,
                              FilenameValidator, IntegerValidator, Spinner)
from jigsaw.io_utils import copy_data_locally, download_data_from_s3
from jigsaw.models.mask.filtering import (ingest_metadata, load_metadata, and_filter, or_filter, join_sets)
from jigsaw.models.mask.transforms import (load_labels, perform_transforms, Transform)


class LabeledImageMask(LabeledImage):
    """Stores pixel-wise-labeled image data and provides related operations

    Attributes:
        image_id (str): the unique ID for the image and labeled data
        image_path (str): the path to the source image
        mask_path (str): the path to the semantic image mask
        label_masks (dict): a dict storing the labels (str) as keys and
            matching pixel colors (3x1 numpy array) in the image mask as values
        xdim (int): width of the image (in pixels)
        ydim (int): height of the image (in pixels)
    """
    _label_to_int_dict = {}

    associated_files = {
        "image": ".png",
        "metadata": "_meta.json",
        "mask": "_mask.png",
        "labels": "_labels.csv"
    }

    training_type = "Semantic Segmentation"

    def __init__(self, image_id, image_path, mask_path, label_masks, xdim,
                 ydim):
        super().__init__(image_id)
        self.image_path = image_path
        self.mask_path = mask_path
        self.label_masks = label_masks
        for label in label_masks:
            self.add_label_int(label)
        self.xdim = xdim
        self.ydim = ydim

    def renumber_label_to_int_dict(self):
        for i, label in enumerate(LabeledImageMask._label_to_int_dict.keys()):
            LabeledImageMask._label_to_int_dict[label] = i + 1

    def delete_label_int(self, label):
        if label in LabeledImageMask._label_to_int_dict.keys():
            del LabeledImageMask._label_to_int_dict[label]
            # renumber all values
            self.renumber_label_to_int_dict()

    def add_label_int(self, label_to_add):
        if label_to_add not in LabeledImageMask._label_to_int_dict.keys():
            # add the new label
            LabeledImageMask._label_to_int_dict[label_to_add] = None

            # renumber all values
            self.renumber_label_to_int_dict()

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

        cwd = Path.cwd()
        data_dir = cwd / 'data'
        mask_filepath = data_dir / str(image_id + "_mask.png")
        mask_filepath = str(
            mask_filepath.absolute())  # cv2.imread doesn't like Path objects.
        labels_filepath = data_dir / str(image_id + "_labels.csv")
        image_filepath = data_dir / str(image_id + ".png")

        labels_df = pd.read_csv(labels_filepath, index_col="label")
        image_mask = cv2.imread(mask_filepath)
        ydim, xdim, _ = image_mask.shape

        label_masks = {}
        for label, color in labels_df.iterrows():
            if label == "Background" and skip_background:
                continue
            color_bgr = np.array([color["B"], color["G"], color["R"]])
            label_masks[label] = color_bgr
        return LabeledImageMask(image_id, image_filepath, mask_filepath,
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
        cwd = Path.cwd()
        data_dir = cwd / 'data'
        labels_filepath = data_dir / str(self.image_id + "_labels.csv")

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
        cwd = Path.cwd()
        data_dir = cwd / 'data'
        mask_filepath = data_dir / str(self.image_id + "_mask.png")

        cv2.imwrite(str(mask_filepath.absolute()), changed_mask)

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
        image_format = b'png'

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
        for label_name, label_int in cls._label_to_int_dict.items():
            label_info = "\n".join([
                "item {", "  id: {id}".format(id=label_int),
                "  name: '{name}'".format(name=label_name), "}"
            ])
            label_map.append(label_info)
        with open(label_map_filepath, 'w') as outfile:
            outfile.write("\n\n".join(label_map))
