# PolNet-Synaptic


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/anmartinezs/polnet/blob/main/LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/anmartinezs)


Python package for generating synthetic datasets of the cellular context for Cryo-Electron Tomography.

## Installation

### Requirements

- **IMOD** must be installed on the system since PolNet calls to some of its standalone commands: https://bio3d.colorado.edu/imod/doc/guide.html
- Miniconda or Anaconda with Python 3.
- Git.
- IMOD can be used for MRC files visualization. Paraview can be used for VTK (.vtp) files visualization. Pandas is recommended for managing the CSV files.

### Installation procedure
Here is how to get it installed:

1. Download PolNet source code:
    ```console
    git clone https://github.com/anmartinezs/polnet.git
    cd polnet
    ```
2. Create a conda virtual environment
    ```console
    conda create --name polnet pip
    conda activate polnet
    ```

3. Install PolNet package with its requirements:
    ```console
    pip install -e .
    ```
**For developers** who do not want to install PolNet in the virtual environment as a package, you can only install
the requirements by:

    pip install -r requirements.txt

You can check all requirements in the **requirements.txt** file (JAX is optional).

The installation has been tested in Ubuntu 22.04 and Windows 10 and 11.

## Usage

Perfect — here’s a **GitHub-ready Markdown** version of your README section.
You can copy-paste this directly into your `README.md` file, and it will render correctly on GitHub with proper headings, indentation, and code blocks:

---

## 🧬 All Features Tomogram Simulation Script

This repository contains a **Python script for generating tomograms** simulating various features such as membranes, helicoidal fibers, globular protein clusters, and more.
The script is **highly configurable** and allows users to specify parameters via **command-line arguments**.

---

### 🚀 Features

**Simulates tomograms with:**

* Membranes *(spherical, elliptical, toroidal)*
* Helicoidal fibers *(actin, microtubules)*
* Globular protein clusters
* Membrane-bound proteins

**Outputs:**

* Simulated density maps (`.mrc`)
* Polydata files (`.vtp`)
* STAR file mapping particle coordinates and orientations with tomograms
* Configurable via command-line arguments
* Includes logging and input file previews

---

### ⚙️ Usage

#### 1 Run the Script with Default Parameters

```console
python all_features_argument.py
```

#### 2 Run the Script with Custom Output Directory

```console
python all_features_argument.py --out_dir /path/to/output
```

#### 3 Run the Script with Custom VOI Shape and Number of Tomograms

```console
python all_features_argument.py --voi_shape 1024 1024 250 --ntomos 10
```

#### 4 Run the Script with Synaptic Settings

```console
python all_features_short_sn_parallel.py --out_dir /path/to/output
```
#### 5 Preview Input Files

The script automatically **previews input files** and saves them as **`.tar.gz` archives** in the output directory.

---

### 📦 Output

#### 🧪 Generated Files

* Simulated density maps (`.mrc`)
* Polydata files (`.vtp`)
* STAR file mapping particle coordinates and orientations with tomograms

#### 🧾 Log Files

* `simulation-output_<job_id>.log` — General log messages
* `simulation_<job_id>_error.log` — Error messages

#### 📊 Statistics

The script generates detailed statistics for each simulated tomogram, including:

* Number of membranes, actin, microtubules, proteins, and membrane-bound proteins
* Volume occupied by each feature
* Total time taken for the simulation

---
    
## Documentation

Folder **docs** contains the file **default_settings.pdf**, it describes the defaults settings for the hardcoded script to generate synthetic tomogram **scripts/data_gen/all_features.py**.

In addition, table in **docs/molecules_table.md** contains more detailed descriptions of the PDB models used to create macromolecular models provided in **data** folder.

## For developers

### Package description
* **polnet**: python package with the Python implemented functionality for generating the synthetic data.
* **gui**: set of Jupyter notebooks with Graphic User Interface (GUI).
  * **core**: functionality required by the notebooks.
* **scripts**: python scripts for generating different types of synthetic datasets. Folders:
  + **data_gen**: scripts for data generation.
    * **deprecated**: contains
    some scripts for evaluations carried out during the software development, they are not prepared for external users
    because some hardcoded paths need to be modified.
      * **templates**: scripts for building the structural units for macromolecules (requires the installation **EMAN2**). Their usage is strongly deprecated, now GUI notebooks include all functionality.
  + **csv**: scripts for postprocessing the CSV generated files.
  + **data_prep**: script to convert the generated dataset in [nn-UNet](https://github.com/MIC-DKFZ/nnUNet) format.
* **tests**: unit tests for functionalities in **polnet**. The script **tests/test_transformations.py** requires to generate at
least 1 output tomo with the script **scripts/all_features.py** and modified the hardcoded input paths, that is because
the size of the input data avoid to upload them to the repository.
* **data**: contains input data, mainly macromolecules densities and configuration input files, that con be used to simulate tomograms. These are the default input, an user can add/remove/modify these input data using the notebooks in **GUI**.
  * **in_10A**: input models for macromolecules at 10A voxel size.
  * **in_helix**: input models for helical structures.
  * **in_mbsx**: input models for membrane structures.
  * **tempaltes**: atomic models and density maps used by macromolecular models.
* **docs**:
  * API documentation.
  * A PDF with the suplementary material for [1] with the next tables:
    + Glossary of acronyms by order of appearance in the main text.
    + Glossary mathematical symbols defined in the main text organized by scope
    + Table Variables used by the input files to model the generators.
    + Table with the structures used to simulate the cellular context.

### Code documentation

The API documentation for polnet Python package is available in [docs/apidoc/index.html](http://htmlpreview.github.io/?https://raw.githubusercontent.com/anmartinezs/polnet/main/docs/apidoc/index.html)


## Main publication (Citation)

[1] Martinez-Sanchez A.*, and Lamm L., Jasnin M. and Phelippeau H. (2024) "Simulating the cellular context in synthetic datasets for cryo-electron tomography" *IEEE Transactions on Medical Imaging* [10.1109/TMI.2024.3398401](https://doi.org/10.1109/TMI.2024.3398401)




