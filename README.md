# wine-machine-learning

A repository containing the code and documentation for the course project on CS-C3240 Machine Learning at Aalto University

## Dependencies

This project uses multiple libraries for data handling and visualization with Python.
They can be installed on Ubuntu based systems with the following commands using apt

```bash
sudo apt update
sudo apt install python3-numpy python3-pandas python3-sklearn python3-matplotlib python3-seaborn
```

If running this inside a virtual environment, use these instead:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Usage

The repository currently only contains a preprocessing script (inside `src`). This can be run with the following command:

```bash
python3 preprocess.py
```

## Files

`doc` Contains the documentation for the project

`src` Contains the source code used for the project

## Copyright

This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.
The original dataset is available here <https://archive.ics.uci.edu/dataset/186/wine+quality>.
More information about the license can be found here <https://creativecommons.org/licenses/by/4.0/legalcode>
