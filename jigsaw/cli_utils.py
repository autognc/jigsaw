from __future__ import print_function, unicode_literals

import os
import regex
import sys
import threading
import time

from colorama import Fore
from halo import Halo
from questionary import prompt, Validator, ValidationError


def set_proper_cwd():
    inner_jigsaw_dir = os.path.dirname(os.path.realpath(__file__))
    outer_jigsaw_dir = os.path.dirname(inner_jigsaw_dir)
    os.chdir(outer_jigsaw_dir)


def list_to_choices(l, sort=True):
    """Converts a list of strings to the proper PyInquirer 'choice' format
    
    Args:
        l (list): a list of str values
        sort (bool, optional): Defaults to True. Whether or not the choices
            should be sorted alphabetically.
    
    Returns:
        list: a list of dicts in questionary format
    """
    if sort:
        l = sorted(l)
    choices = []
    for item in l:
        choices.append({"name": item})
    return choices


def user_input(message, default="", validator=None):
    """Prompts the user for input
    
    Args:
        message (str): the message to give to the user before they provide
            input
        default (str, optional): Defaults to "". The default input response
        validator (Validator, optional): Defaults to None. A PyInquirer
            validator used to validate input before proceeding
    
    Returns:
        str: the user's response
    """
    if in_test_mode():
        return default
    if validator is not None:
        question = [
            {
                "type": "input",
                "name": 'value',
                "message": message,
                "default": default,
                "validate": validator,
            },
        ]
    else:
        question = [
            {
                "type": "input",
                "name": 'value',
                "message": message,
                "default": default
            },
        ]
    answer = prompt(question)
    return answer["value"]


def user_selection(message, choices, selection_type="list", sort_choices=True):
    """Prompts the user to select a choice(s) given a message
    
    Args:
        message (str): the message to give to the user before they choose
        choices (list): a list of the options to provide
        selection_type (str, optional): Defaults to "list". Should be "list"
            or "checkbox" for radio-button-style or checkbox-style selections
        sort_choices (bool, optional): Defaults to True. Whether to
            alphabetically sort the choices in the list provided to the user
    
    Returns:
        str or list: the str for the choice selected if "list" is the 
                        selection_type
                     the list of the choices selected if "checkbox" is the 
                        selection_type
    """
    if in_test_mode():
        if selection_type == "list":
            return choices[0]
        elif selection_type == "checkbox":
            return [choices[0]]
    question = [
        {
            "type": selection_type,
            "name": 'value',
            "message": message,
            "choices": list_to_choices(choices, sort_choices)
        },
    ]
    answer = prompt(question)
    return answer["value"]


def user_confirms(message, default=False):
    """Prompts the user to confirm an action by typing y/n
    
    Args:
        message (str): the message to give to the user before they choose
        default (bool, optional): Defaults to False.
    
    Returns:
        bool: the user's response in bool format
    """
    if in_test_mode():
        return default
    question = [
        {
            "type": "confirm",
            "name": "value",
            "message": message,
            "default": default
        },
    ]
    answer = prompt(question)
    return answer["value"]


def in_test_mode():
    try:
        return os.environ["TEST_MODE"]
    except KeyError:
        return False


class Spinner:
    """Wrapper class to prevent bugs with Halo in pytest (see Issue #97)
    """

    def __init__(self, text, text_color):
        self.text = text
        if in_test_mode():
            return
        self._spinner = Halo(text=text, text_color=text_color)

    def start(self):
        if in_test_mode():
            return
        self._spinner.start()

    def succeed(self, text):
        if in_test_mode():
            return
        self._spinner.succeed(text=text)


class FilenameValidator(Validator):
    def validate(self, document):
        ok = regex.match(r"^[0-9a-zA-Z_]+$", document.text)
        if not ok:
            raise ValidationError(
                message='Please enter a valid name (0-9, a-z, A-Z, _)',
                cursor_position=len(document.text))  # Move cursor to end


class IntegerValidator(Validator):
    def validate(self, document):
        try:
            _ = int(document.text)
        except ValueError:
            raise ValidationError(
                message='Please enter a valid integer',
                cursor_position=len(document.text))  # Move cursor to end


class DirectoryPathValidator(Validator):
    def validate(self, document):
        try:
            if not os.path.exists(document.text):
                raise ValidationError(
                    message='The directory at this location does not exist',
                    cursor_position=len(document.text))  # Move cursor to end
        except:
            raise ValidationError(
                message='Please enter a valid directory path',
                cursor_position=len(document.text))  # Move cursor to end
