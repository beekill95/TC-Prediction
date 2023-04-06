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

## Datasets

In order to work with the experiments,
you'll need the [NCEP FNL dataset](https://rda.ucar.edu/datasets/ds083.2/),
which is reanalysis weather data.
To download the data,
first register an account at [Research Data Archive](https://rda.ucar.edu/).

After that, you can execute the script `scripts/download_ncep_fnl.sh`
to download the dataset.
Upon execution, the script will ask for your username and password which you've created at RDA website.
Once the credential is provided,
the script will download data from May to November in every year from 2008 onwards.
Grab ☕ because this is  gonna be long!

The NCEP/FNL dataset is in .grib2 format,
which is not very friendly to work with.
In addition, the dataset contains global data and redundant environment variables
which are not what want.
So we'll execute the script `scripts/extract_environment_features_from_ncep.py`
to extract the domain and variables of interest and convert to netcdf files.

> scripts/extract_environment_features_from_ncep --lat 5 45 --long 100 260 <path_to_ncep_fnl> <output_directory>

This is also a long-running script so make yourself entertained while the script is running.
Moreover, if you have a machine with many cpus, you can speed up the script by providing
`-p <number_of_parallel_processes>` to the script.
The default value is 8 parallel processes.

While the script is doing its job,
you will need a best track data,
which is basically the historical track data of tropical storms.
The data we will be using is IBTrACS, which can be downloaded from [here](https://www.ncei.noaa.gov/products/international-best-track-archive).
You'll be presented with many versions of the best track,
plese use the csv file.

Once the extract script done running,
and the best track data is downloaded,
you can use the following script to create labels for training deep learning models.

> scripts/create_labels_v2
> --best-track <path_to_csv_ibtracs>
> --observations-dir <path_to_extracted_netcdf_output_dir>
> --leadtime <leadtime>

The script will create a `tc_<leadtime>.csv` file in the `<path_to_extracted_netcdf_output_dir>`
containing the labels.

Finally,
use the following script to split the label file into training, validation, and testing.

> scripts/split_train_test_v1
> --test-from <YYYYMMDD: from which the test data starts>
> --val-from <YYYYMMDD: from which the validation data starts>
> --labels <path_to_the_created_label_file>

