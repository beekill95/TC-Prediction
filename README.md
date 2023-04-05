# Tropical Cyclogenesis Forecast using Deep Models

## Dependencies

This project uses a combination of conda and poetry to manage dependencies.
You can use the following command to install the required dependencies:

> conda create -n tc -f environment.yml

Once the installation finishes,
you will need to install poetry.
In order to do that, please refer to [Poetry homepage](https://python-poetry.org/docs/#installation)
for the latest installation method.
Right now, it is:

> curl -sSL https://install.python-poetry.org | python3 -

Finally, to install the remaining dependencies:

> conda activate tc
> poetry install

## Jupyter Notebooks

This project utilizes [Jupytext](https://github.com/mwouts/jupytext)
to save jupyter notebooks as normal python scripts.

Therefore, in order to run the experiments as jupyter notebooks,
you have to open these python scripts as jupyter notebooks.
You can follow the [instruction here](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html#how-to-open-scripts-with-either-the-text-or-notebook-view-in-jupyter)
to do that.

## Folder Structure

The project is organized as followed:

* `scripts/` folder contains both python scripts and bash scripts to download NCEP/FNL data,
pre-process these data into datasets that can be consumed by tensorflow.
* `tc_formation/` folder is the main python module of the project.
In this module, I implement DNN models such as Resnet, Unet,
dataset utilities to load preprocessed datasets,
integrated gradients, etc.
* `experiments/` folder contains the experiments relating to Resnet, Unet, integrated gradients and feature selection.
* `other_experiments/` folder contains my other (crazy) ideas relating to the project.
