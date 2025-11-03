# DS-LINGUISTICS-MEG-MASC

## Description

This project for data science  linguistics.  

Porivdes reading and analysis of Magnetoencephalography (MEG-MASC) data and annotating it with word and phoneme level features. 
These features are then used to perform the decoding analysis and generate figures and results in notebook/analysis.ipynb file.

The decoding is designed to be run on HPC slurm environment to be tractable. 

## Project Structure

A brief overview of the key directories:

-   **/linguistics/**: Main Python source code for the analysis, data loading, and plotting.
-   **/notebooks/**: Jupyter notebooks for exploratory data analysis.
-   **/scripts/**: Executable scripts for running analyses, including simple analyses, single-subject processing, and Slurm submission scripts.
-   **/data/**: Raw or processed data files used in the analysis.
-   **/output/**: Directory for generated results, such as decoding CSVs and plots.
-   **/test/**: Contains unit and integration tests for the project.
-   **/writing/**: Contains written documents, such as a project proposal and notes.

## Installation

This project uses `uv` for package management and specifies a Python version.

1.  **Install Python:** Ensure you have the correct Python version installed (as specified in the `.python-version` file).
2.  **Install `uv`:** If not already installed, get the `uv` package manager.
3.  **Install Dependencies:** Create the virtual environment and install dependencies using the lock file:
    ```bash
    uv venv
    uv sync
    ```

## Usage

Analyses can be run using the provided scripts.

**Run locally:**
Execute the Python scripts directly:
```bash
source .venv/bin/activate
python scripts/0-simple-analysis.py
python scripts/1-run-single-subject.py --subject 01

