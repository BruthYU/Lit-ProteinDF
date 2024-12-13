<div align="center">


## A Collection of Diffusion Models for Protein Backbone Generation based on Pytorch Lightning⚡.


[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/Pytorch-Lightning-yellow)
![Static Badge](https://img.shields.io/badge/Config-Hydra-blue)
![](https://img.shields.io/badge/PRs-Welcome-green)
![GitHub Repo stars](https://img.shields.io/github/stars/BruthYU/Lit-ProteinDF?style=social)
</div>

## Integrated Methods
| **Name**  | **Paper**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | **Venue** |  **Date**  |                                                                               **Code**                                                                               |
|-----------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| FrameDiff | ![Star](https://img.shields.io/github/stars/jasonkyuyim/se3_diffusion?style=social) <br> [**SE(3) diffusion model with application to protein backbone generation**](https://openreview.net/forum?id=m8OUBymxwv) <br>                                                                                                                                                                                                                                                                                         |   ICML    | 2023-04-25 |                                                        [Github](https://github.com/jasonkyuyim/se3_diffusion)                                                        |
| FoldFlow  | ![Star](https://img.shields.io/github/stars/DreamFold/FoldFlow?style=social&label=Star) <br> [**SE(3)-Stochastic Flow Matching for Protein Backbone Generation**](https://openreview.net/forum?id=kJFIH23hXb) <br>                                                                                                                                                                                                                                                                                            |   ICLR    | 2024-04-21 |                                                           [Github](https://github.com/DreamFold/FoldFlow)                                                            | 
| Genie2    | ![Star](https://img.shields.io/github/stars/aqlaboratory/genie2?style=social&label=Star) <br> [**Out of Many, One: Designing and Scaffolding Proteins at the Scale of the Structural Universe with Genie 2**](https://arxiv.org/abs/2405.15489) <br>                                                                                                                                                                                                                                                          |   arxiv   | 2024-05-24 |                                                           [Github](https://github.com/aqlaboratory/genie2)                                                           |
| FrameFlow | ![Star](https://img.shields.io/github/stars/microsoft/protein-frame-flow?style=social&label=Star) <br> [**Improved motif-scaffolding with SE(3) flow matching**](https://openreview.net/forum?id=fa1ne8xDGn) <br>                                                                                                                                                                                                                                                                                             |   TMLR    | 2024-07-17 |                                                   [Github](https://github.com/microsoft/protein-frame-flow)                                                          |

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
### How to Preprocess Dataset and Build Cache
<details>

All preprocess operations (i.e. how pdb files map to the lmdb cache) are implemented in the folder `Lit-ProteinDF/preprocess`. Please refer to this [README.md](preprocess/README.md) for more instructions. 

Lit-ProteinDF featurizes proteins with the [Alphafold Protein Data Type](https://github.com/google-deepmind/alphafold/blob/d95a92aae161240b645fc10e9d030443011d913e/alphafold/common/protein.py), and build `lmdb` cache following the [FoldFlow](https://github.com/DreamFold/FoldFlow/blob/20abc40dc241bbed408c5aa35a2a39b7778d6372/foldflow/data/pdb_data_loader.py#L323) method.
Different protein files (`mmcif, pdb and jsonl`) are unifed into one data type, thus the built cache could be loaded for all integrated methods during training.
```sh
python preprocess/process_pdb_dataset.py
# Intermediate pickle files are generated.
python preprocess/build_cache.py
# Filtering configurations are listed in config.yaml, the lmdb cache will/should be placed in preprocess/.cache. 
```


**You can also directly download our preprocessed dataset**: [Coming Soon]

</details>

### How to Run Training and Inference

<details>

Training and inference of all integrated methods are implemented in the lightning workspace (`Lit-ProteinDF\lightning`). You can refer to this  [README.md](lightning/README.md) for more details.


</details>

### How to Evaluate Different Methods
<details>
</details>