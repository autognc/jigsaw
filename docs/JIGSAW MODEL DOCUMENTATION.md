##JIGSAW MODEL DOCUMENTATION

### Introduction

Jigsaw is a dataset creation tool. For more information on Jigsaw see the [documentation](https://github.com/autognc/jigsaw/blob/master/README.md). This documentation will cover jigsaw model creation. A Jigsaw model is created in order to specify a specific type or style of dataset creation.  The jigsaw CLI then allows the user to decide which model they would like to use to create their dataset. It is up to the user to decide which model best fits their input data and analysis goals. 



Jigsaw models generally contain the following features:

- Importing data locally or from AWS
- Exporting dataset in TFRecord format locally or to AWS
- Filtering data in the dataset 
  - Choosing specific labels or categories to include or exclude from the dataset
- Transforming datasets
  - Editing and merging categories of the dataset 



Although the above functionality is typical of current models, there is significant flexibility for the model developer to add or omit functionality, or change the way features are implemented.



This documentation will cover how to create a basic Jigsaw model. 



### File Structure

```
jigsaw/											# outer jigsaw directory
	jigsaw/										# inner jigsaw directory
		models/									# stores all jigsaw models
			model_name/						# named after user-created model
				__init.py__					# indicates python module
				model.py						# main module of model
				filtering.py				# optional, for filtering functionality
				transforms.py				# optional, for transform fuctionality
		data_models.yml					# YAML file containing all model name	
				
```



This is the typical file structure of a jigsaw model. All relevant files are included. These files will be covered in depth below, part by part. 

###Files 

###`model.py` 

The model.py module contains the majority of model creation functionality. It must contain a subclass of the `LabeledImage` abstract class and implement the methods found in that class. 

```python
from jigsaw.data_interface import LabeledImage

class LabeledImageSample(LabeledImage):
	#implement all required methods
```

We will now go over the required methods.

######`__init__`

```python
def __init__(self, sample_attribute1, sample_attribute2):     
  # Instantiate instance variables
  self.sample_attribute1 = sample_attribute1
  self.sample_attribute2 = sample_attribute2
  
```

The attributes passed into the constructor vary from model to model, but should contain all necessary information for dataset creation and later data analysis and training. 

Attributes generally include: image_id, image_path, image_dimensions, label_masks and more. 

These attributes are defined in the [`construct`](#`construct`) method.

###### `construct`

```python
@classmethod
def construct(cls, image_id, **kwargs):
  # calculate and define attributes to be passed to the constructor
  return LabeledImageSample(sample_attribute1,sample_attribute2)
```

This method is an alternative constructor. It is called with the argument `image_id`, a unique ID for an image. (This ID is generated in the [`load_metadata`](#load_metadata) method of the [`filtering.py`](#filtering.py) module).

###### `filter_and_load`

```python
	 @classmethod
   def filter_and_load(cls, data_source, **kwargs):
        ingest_metadata(data_source, **kwargs)
				
        tags_df = load_metadata() #pandas dataframe containing image_ids and labels
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

```

The `filter_and_load` method takes in the argument `data_source` (either "Local" or "S3") and a data_filepath or bucket name pointing to the relevant image data. The method allows users to specify the categories of data included in the dataset. The method returns `image_ids` (a dataframe containing desired image IDs and labels), and `filter metadata` (a dictionary detailing the number of images of each category contained in the dataframe)

The above code has been implemented identically across all current models and it is likely that the `filter_and_load` method as shown here can be implemented without changes into future models. However, changes can be made to the method if required.

Note that there are method calls to methods found in the `filtering.py` module (`ingest_metadata`,` load_metadata`, `and_filter`, `or_filter`,` join_sets`). See [`filtering.py`](#filtering.py).



######`transform`

```python
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

```



The `transform` method takes the argument `image_ids` (the dataframe of image IDs and labels) and returns `tranforms_metadata` (a dictionary detailing any transformations done to the dataset).  The method allows users to make changes to the dataset, such as editing labels or merging categories. 

As with [`filter_and_load`](#filter_and_load), it is likely that one may implement this method as is, without making any changes to the code. However, changes may be made if desired. 

Note that there are method calls to methods found in the `transforms.py` module (` load_labels`,``Transforms`,` perform_transforms`). See [`transforms.py`](#transforms.py).



###### `export_as_TFExample`

```python
def export_as_TFExample(self,**kwargs):
  
  # define the features that must be included in the TFExample
  sample_feature1 = self.sample_attribute1
  sample_feature2 = self.sample_attribute2
  
  #create tf_example 
  tf_example = tf.train.Example(
    features=tf.train.Features(
      feature={
        'sample_feature1':
        dataset_util.int64_feature(sample_feature1),
        'sample_feature2':
        dataset_util.int64_feature(sample_feature2),
        }))
  
  return tf_example
```

The `export_as_TFExample` method converts a `LabeledImageSample` object to tf_example (tf.train.Example) and returns this tf_example. 

When creating this tf_example, it is important to consider what features will be necessary in the training stage, and ensure that these features are included. 

For more on the tf.Example format, consult the [TensorFlow Documentation](https://www.tensorflow.org/tutorials/load_data/tf_records).



######`write_additional_files`

```python
@classmethod
def write_additional_files(cls, dataset_name, **kwargs):
  # add any other necessary files to the dataset directory
     
```

The `write_additional_files` method allows takes in the name of the dataset. It is a flexible method that should be used for adding any necessary files to the dataset that have not yet been added. One possible use of the method would be to write out the TensorFlow Object Detection Label Map, as is seen in several current models. 



### `filtering.py`

This module offers methods to assist in the filtering done by [`filter_and_load`](#`filter_and_load`). 



###### `ingest_metadata`

```python
def ingest_metadata(data_source, **kwargs):
    only_json_func = lambda filename: filename.endswith("_meta.json")

    spinner = Spinner(text="Loading metadata...", text_color="magenta")
    spinner.start()

    if data_source == "Local":
        copy_data_locally(
            source_dir=kwargs["data_filepath"], condition_func=only_json_func)
    elif data_source == "S3":
        download_data_from_s3(
            bucket_name=kwargs["bucket"], condition_func=only_json_func)

    spinner.succeed(text=spinner.text + "Complete.")
```

The `ingest_metadata` method takes in the argument `data_source`, a string specifying where image data is stored. The method then either copies or downloads this data. 

###### `load_metadata`

```python
def load_metadata():
   
    tags_df = pd.DataFrame()
    cwd = Path.cwd()
    data_dir = cwd / 'data'

    os.chdir(data_dir)

    # parse data_dir to populate dataframe with image_ids and labels
    
    os.chdir(cwd)

    return tags_df
```

The `load_metadata` method takes no arguments and returns the `tags_df` pandas DataFrame that is used elsewhere in the model. 

The format of the DataFrame is as follows:

		- rows: contains image ID strings
		- column headers: each tag (category)
		- columns: True/False values indicating whether the image has a tag in that column header

The model maker must parse the data in data_dir to find this information for each image and populate the DataFrame.



###### `and_filter`

```python
def and_filter(tags_df, filter_tags):
  
    subset = tags_df
    for tag in filter_tags:
        subset = subset[subset[tag]]
    return subset

```

The `and_filter` method method takes in as arguments the `tags_df` [DataFrame](#`load_metadata') and `filter_tags`, a list of string values of which tags should be used in applying the filter. The method returns `subset`, a DataFrame formatted identically to `tags_df`, but containing only the images that passed the AND filter. 

This method filters out a set of images based on the *intersection* of its tag values. With this filter, an image must possess all tags in the `filter_tags` list to pass the filter.



###### `or_filter`

```python
def or_filter(tags_df, filter_tags):

    result = pd.DataFrame()
    for tag in filter_tags:
        subset = tags_df[tags_df[tag]]
        result = pd.concat((result, subset), sort=False)
    result = result.reset_index().drop_duplicates(
        subset='index', keep='first').set_index('index')
    return result

```

The `or_filter` method method takes in as arguments the `tags_df` [DataFrame](#`load_metadata') and `filter_tags`, a list of string values of which tags should be used in applying the filter. The method returns `result`, a DataFrame formatted identically to `tags_df`, but containing only the images that passed the OR filter. 

This method filters out a set of images based on the *union* of its tag values. With this filter, an image must posses at least one tag in the `filter_tags` list in order to pass the filter. 



`join_sets`

```python
def join_sets(sets):

    result = pd.DataFrame()
    for group in sets:
        result = pd.concat((result, group), sort=False)
    result = result.reset_index().drop_duplicates(
        subset='index', keep='first').set_index('index')
    return result
```

The `join_sets` method takes in a list of datasets that should be merged into on (the union of all datasets); each element in this list is a DataFrame in the format of the `tags_df` [DataFrame](#`load_metadata'). The method returns a DataFrame, of the same format as the input DataFrames, representing the union of all of the input DataFrames with duplicates removed.

This method returns the unions of a set of datasets and removes duplicates. 



###`data_models.yml`

Every model must be added to the `data_models.yml` YAML file. Each addition to this file should be formatted as follows:

```yaml
- parent_module: jigsaw.models.[MODEL_NAME].model  # replace [] with name of model
  model_class: [MODEL_LABELEDIMAGE_CLASS] # replace [] with name of class

```

The `model_class` field requires the name of the class created in the new model that inherits from the [`LabeledImage`](#`model.py`) class. 

The `parent_module` field requires the name of the module that contains `model_class`