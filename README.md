## Installation
PCDR works with the following operating systems:

* Linux
* Windows 10
* macOS X

PCDR requires Python version 3.7 or later.

PCDR requires torch version 1.7.0 or later. If you want to use PCDR with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install 
```bash
pip install -e . --verbose
pip install -r requirements.txt
pip install ray[tune]
```

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole.py
```

This script will run the PCDR model on the netflix dataset.


If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run_recbole.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_recbole.py --model=[model_name]
```

If you want to change the dataset, just run the script by setting additional command parameters:

```bash
python run_recbole.py --dataset=[dataset_name]
```