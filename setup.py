from setuptools import setup


def dependencies(file):
    with open(file) as f:
        return f.read().splitlines()


setup(
    name='jigsaw',
    version='0.0.1',
    description='Dataset Preparation/Engineering Tool',
    packages=['jigsaw'],
    install_requires=dependencies('requirements.txt'))