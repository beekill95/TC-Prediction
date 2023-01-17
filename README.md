# Tropical Cyclogenesis Forecast using Deep Models

## Dependencies

This project uses conda to manage dependencies.
You can use the following command to install the required dependencies:

> conda create -n tc -f environment.yml

After the installation finishes,
you can activate the environment `tc` to run the experiments in this project.

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
