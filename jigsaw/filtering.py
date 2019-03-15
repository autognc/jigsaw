import json
import os

import pandas as pd

from pathlib import Path
from io_utils import download_json_metadata_from_s3


def load_metadata():
    """Downloads all image metadata JSONs and loads their tags
    
    Returns:
        DataFrame: a pandas DataFrame storing image IDs and associated tags;
            index (rows) = image ID (str)
            column headers = the tags themselves
            columns = True/False values for whether the image has the tag in
                in that column header
    """
    tags_df = pd.DataFrame()
    download_json_metadata_from_s3()
    cwd = Path.cwd()
    data_dir = cwd / 'data'
    json_dir = data_dir / 'json'

    os.chdir(json_dir)
    for dir_entry in os.scandir():
        if not dir_entry.name.endswith("meta.json"):
            continue
        image_id = dir_entry.name.rstrip("_meta.json")
        with open(dir_entry.name, "r") as read_file:
            data = json.load(read_file)
        tag_list = data["tags"]
        temp = pd.DataFrame(
            dict(zip(tag_list, [True] * len(tag_list))), index=[image_id])
        tags_df = pd.concat((tags_df, temp), sort=False)
    tags_df = tags_df.fillna(False)
    os.chdir(cwd)

    return tags_df


def and_filter(tags_df, filter_tags):
    """Filters out a set of images based upon the intersection of its tag values
    
    NOTE: With an AND (intersection) filter, an image must possess all tags in
    the `filter_tags` list in order to pass the filter.

    Args:
        tags_df (DataFrame): a pandas DataFrame storing image IDs and
            associated tags; its structure is:
                index (rows) = image ID (str)
                column headers = the tags themselves
                columns = True/False values for whether the image has the tag
                    in that column header
        filter_tags (list): a list of str values for the tags that should be
            used to apply the filter
    
    Returns:
        DataFrame: a subset of the input DataFrame with those images that
            passed the AND filter remaining
    """
    subset = tags_df
    for tag in filter_tags:
        subset = subset[subset[tag]]
    return subset


def or_filter(tags_df, filter_tags):
    """Filters out a set of images based upon the union of its tag values
    
    NOTE: With an OR (union) filter, an image must possess at least one tag in
    the `filter_tags` list in order to pass the filter.

    Args:
        tags_df (DataFrame): a pandas DataFrame storing image IDs and
            associated tags; its structure is:
                index (rows) = image ID (str)
                column headers = the tags themselves
                columns = True/False values for whether the image has the tag
                    in that column header
        filter_tags (list): a list of str values for the tags that should be
            used to apply the filter
    
    Returns:
        DataFrame: a subset of the input DataFrame with those images that
            passed the OR filter remaining
    """
    result = pd.DataFrame()
    for tag in filter_tags:
        subset = tags_df[tags_df[tag]]
        result = pd.concat((result, subset), sort=False)
    result = result.reset_index().drop_duplicates(
        subset='index', keep='first').set_index('index')
    return result


def join_sets(sets):
    """Returns the union of a set of datasets
    
    NOTE: this function removes duplicates, so it is possible to end up with a
    smaller number of items than are given as inputs.

    Args:
        sets (list): a list of datasets that should be merged into one (the
            union of all datasets); each element of the list is a DataFrame
            in the format:
                index (rows) = image ID (str)
                column headers = the tags themselves
                columns = True/False values for whether the image has the tag
                    in that column header
    
    Returns:
        DataFrame: a DataFrame that represents the union of all of the input
            DataFrames with duplicates removed; it stores image IDs and 
            associated tags in the format:
                index (rows) = image ID (str)
                column headers = the tags themselves
                columns = True/False values for whether the image has the tag
                    in that column header
    """
    result = pd.DataFrame()
    for group in sets:
        result = pd.concat((result, group), sort=False)
    result = result.reset_index().drop_duplicates(
        subset='index', keep='first').set_index('index')
    return result