<div align="center">


## A Collection of Protein Structure Generation Diffusion Models bases on Pytorch Lightning⚡.


[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/last_commit-Nov-blue)
![](https://img.shields.io/badge/PRs-Welcome-yellow)
![GitHub Repo stars](https://img.shields.io/github/stars/BruthYU/Lit-ProteinDF?style=social)
</div>

### Integrated Methods
| **Name**  | **Paper**                                                                                                                                                                                                                                            | **Venue** |  **Date**  |                                                  **Code**                                                   |
|-----------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------:|:----------:|:-----------------------------------------------------------------------------------------------------------:|
| FrameDiff | ![Star](https://img.shields.io/github/stars/jasonkyuyim/se3_diffusion?style=social) <br> [**SE(3) diffusion model with application to protein backbone generation**](https://openreview.net/forum?id=m8OUBymxwv) <br>                                          |   ICML    | 2023-04-25 |                           [Github](https://github.com/jasonkyuyim/se3_diffusion)                            |
| FoldFlow  | ![Star](https://img.shields.io/github/stars/DreamFold/FoldFlow?style=social&label=Star) <br> [**SE(3)-Stochastic Flow Matching for Protein Backbone Generation**](https://openreview.net/forum?id=kJFIH23hXb) <br>                                             |   ICLR    | 2024-04-21 |                               [Github](https://github.com/DreamFold/FoldFlow)                               | 
| Genie2    | ![Star](https://img.shields.io/github/stars/aqlaboratory/genie2?style=social&label=Star) <br> [**Out of Many, One: Designing and Scaffolding Proteins at the Scale of the Structural Universe with Genie 2**](https://arxiv.org/abs/2405.15489) <br> |   arxiv   | 2024-05-24 |                               [Github](https://github.com/aqlaboratory/genie2)                               |

### Installation

**Note: Please use Python 3.9+ for Lit-ProteinDF**

To get started, simply create conda environment and run pip installation:

```shell
conda create -n Lit-ProteinDF python=3.9
git clone https://github.com/BruthYU/Lit-ProteinDF
...
cd Lit-ProteinDF
pip install -r requirements.txt
```


### Use Lit-ProteinDF