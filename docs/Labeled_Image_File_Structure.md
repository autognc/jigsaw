# Labeled Input Image File Structure for Use with Jigsaw

Images in an S3 bucket should be organized in the following way:
```
<prefix>                    # useful prefix that describes the images it contains
    <image name>.png            # image itself
    <image name>_labels.csv     # semantic labels
    <image name>_mask.png       # semantic mask
    <image name>_meta.json      # metadata
    ... for all images
... for all prefixes
```

A prefix is similar to a directory. The difference is that S3 is a flat file structure 
and does not actually have any concept of folders/directories. Prefixes are simply
placed before a filename to mimic directory structure and are displayed as directories 
on the S3 GUI. They offer a way to organize data within a bucket.

Jigsaw will first present the user with a choice for which prefixes they desire to 
create a dataset from. This happens *without* downloading any image metadata. Once 
prefixes are selected, metadata for all images in the selected prefixes is downloaded
and dataset creation continues as normal.