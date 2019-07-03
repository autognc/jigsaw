
**<span style="text-decoration:underline;">Jigsaw Overview</span>**


### Entry Point:


#### jigsaw/cli.py: Asks users questions and acts on responses.



1. Model Selection: Lists available custom models.
2. Local/S3 Data
    1. Local: Asks for path to dir.
    2. S3: Uses LABELED_BUCKET_NAME env variable.
3. Asks if you’d like to filter (conditional and/or/join) any of the step 2 data.
4. Asks if you’d like to perform any transforms (rename/merge) on data from step 3.
5. Name the output dataset.
6. Specify number of folds to construct.
7. Add notes/user’s name.
8. Writes the data locally (write_dataset and write_metadata).
9. Confirm if data should be uploaded to S3.


### Helpers:


#### jigsaw/bounding_box.py: LOOK TO REFACTOR. Old version of BBoxLabeledImage before Gavin’s refactor.



*   Contains class for BBoxLabeledImage. This class relates labeled bounding boxes for an image, and a label for what’s within the box (human, dog, Cygnus, etc.). Main entry point is from_labeled_image_mask, which takes a labeled_image_mask, extracts labeled bounding boxes from the pixel-wise labels, and returns a BBoxLabeledImage object. Also contains class for BoundingBox, a member of the BBoxLabeledImage class.
    *   class BBoxLabeledImage
        *   __init__
        *   from_labeled_image_mask(...) ← Main entry point.
        *   convert_to_tf_example(...) ← Important for TensorFlow.
        *   convert_to_dict(...)
    *   class BoundingBox
        *   __init__(...)
        *   __repr__(...)
        *   convert_to_dict(...)


#### jigsaw/models/bbox/model.py: New version of BBoxLabeledImage after refactor.



*   Gavin re-wrote the original custom model to conform to jigsaw/data_interface.py. Same functionality should exist as in the previous version, plus more.


#### jigsaw/cli_utils.py: Used throughout cli.py flow’s interface.



*   Pretty well documented by Gavin. Contains helpers for the cli.py flow. This includes validators, Spinner class, user confirmation prompts, user list selection, and a function to change the run directory for referencing absolute file paths.
    *   set_proper_cwd()
    *   list_to_choices(...)
    *   user_input(...)
    *   user_selection(...)
    *   user_confirms(...)
    *   int_test_mode()
    *   class Spinner
        *   __init__/start/succeed
    *   FilenameValidator(...)
    *   IntegerValidator(...)
    *   DirectoryPathValidator(...)


#### jigsaw/data_interface.py:



*   Interface for creating new, custom data types for use with Jigsaw. Interfaces in Python are a rather peculiar topic. You’ll need to examine the imports used to achieve this.
    *   class LabeledImage(ABC)
        *   __init__(...)
        *   construct(...)
        *   filter_and_load(...)
        *   transform(...)
        *   associated_files(...)
        *   training_type(...)
        *   export_as_TFExample(...)
        *   write_additional_files(...)
        *   construct_all(...)
        *   copy_associated_files(...)
    *   load_models()


#### jigsaw/data_models.yml: Provides data for Step 1.



*   Lists the custom data types made for use with jigsaw. (As of writing, just LabeledImageMasks and BBoxLabeledImages.)


#### jigsaw/filtering.py: Step 3 from flow.



*   Heavy lifting for the filtering of dataset items that you’d like to pass to the pipeline. You’ll have to examine the code more closely for the and/or/join logic.
    *   load_metadata()
    *   and_filter(...)
    *   or_filter(...)
    *   join_sets(...)


#### jigsaw/io_utils.py: Steps 2/9 from flow.



*   Handles downloading/uploading dataset from/to S3/locally. 
    *   download_image_and_save(...)
    *   load_remote_image(...)
    *   copy_data_locally(...)
    *   download_data_from_s3(...)
    *   download_json_metadata_from_s3(...)
    *   download_image_data_from_s3(...)
    *   get_s3_filepath(...)
    *   load_BBoxLabeledImages(...)
    *   load_LabeledImageMasks(...)
    *   upload_dataset(...)


#### jigsaw/mask.py: LOOK TO REFACTOR.



*   Contains class for LabeledImageMask. This relates an image to pixel-wise information about what object a given pixel is a part of.
    *   class LabeledImageMask
        *   __init__
        *   renumber_label_to_int_dict(...)
        *   delete_label_int(...)
        *   add_label_int(...)
        *   from_files(...) ← Main entry point.
        *   rename_label(...)
        *   merge_labels(...)
        *   save_label_changes(...)
        *   save_mask_changes(...)
        *   convert_to_tf_example(...) ← Import for tensorflow.


#### jigsaw/models/mask/model.py: New version of LabledImageMask after refactor.



*   Gavin re-wrote the original custom model to conform to jigsaw/data_interface.py. Same functionality should exist as in the previous version, plus more.


#### jigsaw/transforms.py: Step 4 from flow.



*   In Step 4, data can be transformed via renames/merges. Gavin made it so these actions could be represented as objects, and therefore be performed all at once in a multi-threaded manner to limit I/O costs. 
    *   load_labels(...)
    *   perform_transforms(...)
    *   class Transform


#### jigsaw/write_dataset.py: Step 6 from flow.



*   write_dataset(...)
    1. Deletes output dataset if namespace already in use. _delete_dir(...)_
    2. Splits data into test/dev. _split_data(...)_
    3. Writes related data (any supported data items like images).
    4. Divides dev set into folds._ divide_into_folds(...)_
    5. Writes out each fold. _write_out_fold(...)_