# SMaTD

`SMaTD` (**S**urrogate **Ma**chine **T**ranslation **D**etection) is a tool implemented in python that leverages the internal representations of a machine translation (MT) system to determine, given a source sentence, whether a translation is human- or machine-generated.

In this repository, you will find the code, along with the dataset and the models trained during our research, available in the [releases section](../../releases).

## Installation

To install `SMaTD`, first clone the code from the repository:

```bash
git clone https://github.com/transducens/SMaTD.git
```

Create a conda environment to isolate the python dependencies and install pytorch:

```bash
conda create -n smatd -c conda-forge python==3.11.9
conda activate smatd
# Follow https://pytorch.org/get-started/locally/ to install pytorch 2.4.0
```

Install `SMaTD`:

```bash
cd SMaTD

pip3 install .
```

Check out the installation:

```bash
# Usage

smatd --help
smatd_lm_baseline --help
```

## Usage

Some scripts require pickle files, but generating sentence representations on the fly is also supported. Use `smatd/nllb_get_log_prob.py` to create these files. This consumes more disk space but greatly reduces time when training for multiple epochs.

Check the `--help` flag for `smatd` and `smatd_lm_baseline` to see the available configuration options. The experiment scripts may also serve as a good starting point.

## Citation

If you use SMaTD or the resources provided in this repository, please cite our work as follows:

TODO

## Acknowledgements

This paper is part of the work conducted in R+D+i projects PID2021-27999NB-I00 and PID2024-158157OB-C31 funded by the Spanish Ministry of Science and Innovation (MCIN), the Spanish Research Agency (AEI/10.13039/501100011033) and the European Regional Development Fund A way to make Europe. Cristian García-Romero is funded by Generalitat Valenciana and the European Social Fund [reserach grant CIACIF/2021/365]. Some of the computational resources used were funded by the Valencia Government and the European Regional Development Fund (ERDF) through project IDIFEDER/2020/003.

![European Social Fund](img/logo-FSE.jpg)



