# jigsaw
Dataset Preparation/Engineering Tool

## Running Locally
### With Conda
First, make sure you have [Anaconda](https://www.anaconda.com/distribution/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer. You’ll need conda version >=4.6 for environments to work as described below.

Clone this repository to your local machine:
`git clone https://github.austin.utexas.edu/aere-tsl/jigsaw.git`

From within the (outer) jigsaw directory, create a conda environment from the `environment.yml` provided:
`conda env create -f environment.yml`

The environment will contain a Python 3.6 interpreter and all of the dependencies required for Jigsaw. All you need to do is activate it:
`conda activate jigsaw`

Now, you should be able to kick off Jigsaw:
`jigsaw/cli.py`

### Using Docker
TBD…[minconda Docker image](https://hub.docker.com/r/continuumio/miniconda/)
