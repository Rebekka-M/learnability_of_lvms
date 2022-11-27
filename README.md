# Learnability of Lvms
This repository contains the code and examples for the bachelor work *Madsen, Rebekka. "A Numerical Study of Learnability in Latent Variable Models"*.

The Learnability of various LVMs was studied by observing an LVMs ability to represent two clusters of data in latent space so that the two clusters were still separable. The datasets were alterned to contain less or more samples $N$ and have a higher or lower signal-to-noise ratio, $SNR$. 

## Install packages
```bash
pip install -r requirements.txt
```
## Usage 
Shell:
```bash
python from main import *; run() "seed=0|data_dim=1000|SNR=75|n_train=100|dataset_name=default|experiment=(0)"
```

Notebook:
```python
from src.parameters import Parameters
from src.experiment import Experiment

params = {seed:0, data_dim:1000, SNR:75, n_train:100, dataset_name: "default", experiment: "(0)"}
parameters = Parameters(params, mkdir=True)
experiment = Experiment(parameters)

if "audio" in experiment.dataset_name:
    experiment.wav2vec()
    experiment.run_wav2vec_eval()
    
experiment.run_gplvm()
experiment.run_pca()
experiment.run_tsne()
experiment.run_umap()
experiment.run_trimap()
```
