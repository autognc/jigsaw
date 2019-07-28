# Output Dataset File Structure
Terminology:
- **dev**: development data. Serves as a secondary set of validation data for experimentation.
- **train** training data. Split into training and validation data used to evaluate model performance during training.
- **test**: holdout test data. Never looked at and only used for testing the final model.

Dataset outputs from Jigsaw look as follows:
```
<dataset_name>                  # name of dataset
    splits/                     # Set of different splits of the training data
        complete/               # complete set of training images (all folds)
            train/              # contains TF records
        fold_<k>/               # fold used for k-fold validation
            train/              # each fold has an internal train/dev split
            dev/
        standard/               # standard train/dev split of all images
            train/          
            dev/
    test/                       # testing data
    label_map.pbtxt             # TF label map
    metadata.json               # metadata
```

## Scheme
The dataset is first sliced to generate **test** data, usually with an 80/20 split. This 
data is placed into the `test/` directory. The remaining 80% is then split in various 
ways to allow for different styles of training. This data is placed into the `splits/` 
directory.

There are three types of split in `splits/`:
1. `complete/`, which has all non-test images split into train and val sets. You would use this split AFTER performing k-fold validation, or if you simply want to use all available
data for training.
2. `fold_<k>`, a specific fold used for k-fold cross validation. Google it for more information.
3. `standard/`, the standard split of non-test data into train and dev data (defined as above)
with train further split into train/val for the actual training.