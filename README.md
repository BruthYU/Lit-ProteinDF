<div align="center">


## A Collection of Protein Structure Generation Diffusion Models bases on Pytorch Lightning⚡.


[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/Pytorch-Lightning-yellow)
![Static Badge](https://img.shields.io/badge/Config-Hydra-blue)
![](https://img.shields.io/badge/PRs-Welcome-green)
![GitHub Repo stars](https://img.shields.io/github/stars/BruthYU/Lit-ProteinDF?style=social)
</div>

## Integrated Methods
| **Name**  | **Paper**                                                                                                                                                                                                                                            | **Venue** |  **Date**  |                                                  **Code**                                                   |
|-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:----------:|:-----------------------------------------------------------------------------------------------------------:|
| FrameDiff | ![Star](https://img.shields.io/github/stars/jasonkyuyim/se3_diffusion?style=social) <br> [**SE(3) diffusion model with application to protein backbone generation**](https://openreview.net/forum?id=m8OUBymxwv) <br>                                          |   ICML    | 2023-04-25 |                           [Github](https://github.com/jasonkyuyim/se3_diffusion)                            |
| FoldFlow  | ![Star](https://img.shields.io/github/stars/DreamFold/FoldFlow?style=social&label=Star) <br> [**SE(3)-Stochastic Flow Matching for Protein Backbone Generation**](https://openreview.net/forum?id=kJFIH23hXb) <br>                                             |   ICLR    | 2024-04-21 |                               [Github](https://github.com/DreamFold/FoldFlow)                               | 
| Genie2    | ![Star](https://img.shields.io/github/stars/aqlaboratory/genie2?style=social&label=Star) <br> [**Out of Many, One: Designing and Scaffolding Proteins at the Scale of the Structural Universe with Genie 2**](https://arxiv.org/abs/2405.15489) <br> |   arxiv   | 2024-05-24 |                               [Github](https://github.com/aqlaboratory/genie2)                               |

## Installation


To get started, simply create conda environment and run pip installation:

```shell
conda create -n Lit-ProteinDF python=3.9
git clone https://github.com/BruthYU/Lit-ProteinDF
...
cd Lit-ProteinDF
pip install -r requirements.txt
```


## Usage
In this section we will demonstrate how to use Lit-ProteinDF.

---
### Preprocess Dataset and Build Cache
Lit-ProteinDF featurizes proteins using the [Alphafold Protein Data Type](https://github.com/google-deepmind/alphafold/blob/d95a92aae161240b645fc10e9d030443011d913e/alphafold/common/protein.py), and build `lmdb` cache following [FoldFlow](https://github.com/DreamFold/FoldFlow/blob/20abc40dc241bbed408c5aa35a2a39b7778d6372/foldflow/data/pdb_data_loader.py#L323).
In this way, different protein data files (`mmcif, pdb and jsonl`) are unifed into one data type, and the built cache could be loaded for all integrated methods during training.
```sh
python preprocess/process_pdb_dataset.py
# Intermediate pickle files are generated.
python preprocess/build_cache.py
# Filtering configurations are listed in config.yaml, the lmdb cache will/should be placed in preprocess/.cache. 
```
All preprocess operations (i.e. how pdb files map to the lmdb cache) are implemented in the folder `Lit-ProteinDF/preprocess`. Please refer to this [README.md](preprocess/README.md) for more details and instructions. 

**You can directly download our preprocessed cache**: [Coming Soon]