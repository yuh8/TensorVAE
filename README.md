# TensorVAE: A simple and efficient conformation generation model

This is the official implementation of TensorVAE

![alt text](https://github.com/yuh8/TensorVAE/blob/main/assets/TensorVAE.png)

## Conda environment setup

```
# create the env with all dependencies
conda env create -f tensorvae_env.yml
# activate the environment
conda activate tensorvae
```

## Dataset
The official GEOM dataset is accessible from [here](https://github.com/learningmatter-mit/geom)

The qm9 molecular property prediction data is available [here](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv) and [here](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz)

The train, validation, test smiles for reproducing the *Drugs* conformation generation experiment will be available upon acceptance of the paper.

## Training and testing
**Conformation generation**

There are two branchs: **main** for Drugs experiments and **QM9** for qm9 experiments

Please download the geom dataset into a raw_data_path and generate data using

```
python data_gen.py --raw_data_path your/raw_data/path
```

Provide your local train, val and test path to the ```train_test_conf_gen.py``` function

```
python train_test_conf.py --train_path [your/train/path] --val_path [your/val/path] --test_path [your/test/path] 
```

the hyper-parameter setting is available in the ```src.CONSTS.py``` file. Please feel free to tune these parameters. The default hyper-parameters should produce the following training and validation curve

![alt text](https://github.com/yuh8/TensorVAE/blob/main/assets/Train_Val%20curve.png)

For testing, the checkpoints for Drugs experiments will be available upon the acceptance of the paper. Please download and place them in the ./checkpoints/ folder and run

```
python train_test_conf.py --train false --test_path [your/test/path] 
```

This should reproduce the Drugs conformation results presented in Tab.1 of the paper.

Some samples of the generated conformations by the trained Drugs model is shown below

![alt text](https://github.com/yuh8/TensorVAE/blob/main/assets/Generated%20samples.png)

**QM9 molecular property prediction**

The code for molecular property prediction is in the QM9_propert branch.

Please download the QM9 property prediction datasets in your local path according to the path structure in ```train_prop_qm9.py```

Start training by running ```python train_prop_qm9.py```

Start testing by running ```python test_qm9_prop.py```



