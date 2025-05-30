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
python run.py
```

This script will run the PCDR model on the netflix dataset.


If you want to change the parameters, such as ``learning_rate``, ``embedding_size``, just set the additional command
parameters as you need:

```bash
python run.py --learning_rate=0.0001 --embedding_size=128
```

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run.py --model=[model_name]
```
for example:

```bash
python run.py --model=PCDR
```

If you want to change the dataset, just run the script by setting additional command parameters:

```bash
python run.py --dataset=[dataset_name]
```
for example:

```bash
python run.py --dataset=netflix
```

### Download Dataset:
All data sets are stored in the Dataset directory. If not found, please click the link to download, such as
- [amazon-luxury-beauty-18](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon2018/Amazon_Luxury_Beauty.zip)
- [netflix](https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Netflix/netflix.zip)
- More data set downloads can be found at recbole/properties/dataset/url.yaml

### Dataset usage
- Place the downloaded data set in the Dataset directory, and create a separate directory named as the data set, such as
  - dataset/amazon-luxury-beauty-18/amazon-luxury-beauty-18.inter
  - dataset/amazon-luxury-beauty-18/amazon-luxury-beauty-18.item

- Create a yaml file for the data set and store it in the properties directory. You can refer to other data set yaml files when creating a template, for example
  - recbole/properties/dataset/amazon-luxury-beauty-18.yaml
  - recbole/properties/dataset/netflix.yaml

### Drawing code

The directory draw/ stores all the drawing codes of the article
.m files represents matlab code (main)
.py files represents python code (same unction)

### Conservative and Radical Statistics
Add configuration in the data set's yaml file
eg. recbole/properties/dataset/ml-1m.yaml

obtain radicals performance
```
testset_sample_method: radicals
```
obtain conservatives performance
```
testset_sample_method: conservatives
```

If neither is written, it means that there is no distinction between the two.


### Auto-tuning Hyperparameter
Open PCDR/hyper.test and set several hyperparameters to auto-searching in parameter list. The following has two ways to search best hyperparameter:
- **loguniform**: indicates that the parameters obey the uniform distribution, randomly taking values from e^{-8} to e^{0}.
- **choice**: indicates that the parameter takes discrete values from the setting list.
Here is an example for hyper.test:

```bash
learning_rate loguniform -8, 0
embedding_size choice [64, 96 , 128]
train_batch_size choice [512, 1024, 2048]
mlp_hidden_size choice ['[64, 64, 64]','[128, 128]']
```

Set training command parameters as you need to run:
```bash
python run_hyper.py --model=PCDR --dataset=[data_name] --config_files=xxxx.yaml --params_file=hyper.test
e.g.
python run_hyper.py --model=PCDR --dataset=ml-100k --config_files=test.yaml --params_file=hyper.test
```
Note that --config_files=test.yaml is optional, if you don't have any customize config settings, this parameter can be empty.

This processing maybe take a long time to output best hyperparameter and result:

```bash
running parameters:                                                                                                                    
{'embedding_size': 64, 'learning_rate': 0.005947474154838498, 'mlp_hidden_size': '[64,64,64]', 'train_batch_size': 512}                
  0%|                                                                                           | 0/18 [00:00<?, ?trial/s, best loss=?]
```