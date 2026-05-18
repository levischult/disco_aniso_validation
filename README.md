# disco_aniso_validation
Repo containing scripts, notebooks, and datasets for validating anisotropy analyses in discovery

Hello! Welcome and thank you for your interest in helping test the next generation of anisotropy search software! We have a lot to test, so check this spreadsheet and get in touch on the IPTA Slack (#ipta_dr3_anisotropy)

## 1 - installation
1. clone this repo
2. Set up a mamba/virtual environment:
  - environments/jhp_disco.yaml for mamba/conda
  - environments/requirements.txt for python venv / pip
3. Activate the environment
4. Clone Levi's fork of discovery which is the home for the anisotropy tools as of now.
  - Once you have cloned discovery, you will need to navigate into the repo and run `pip install -e .` to install an editable version of discovery.

## 2 - Example Notebooks:
  - examples/ has notebooks to demonstrate how to set up broad/narrowband models.
  - ORF cookbook notebook demonstrates each basis currently implemented.

## 3 - Pick a model to test!
  - There are a multitude of combinations to test! Broad/Narrow and any of the many bases.
  - Contact Levi or Serena to claim!
