## aMMAI Final Project
### Cross-Domain Few-Shot Learning

#### Presentation Slide

[aMMAI 2021 Final Project](https://hackmd.io/@ZBjQiqfLQnSlD3bbbrrIOQ/HJyLO2bs_)

##### The following content is a subset of presentation slides.

#### Performance Table

|           Method           |              FSL              |         CDFSL-Single          |                               |                               |          CDFSL-Multi          |                               |                               |
|:--------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|                            |       mini-<br>ImageNet       |        Crop<br>Disease        |            EuroSAT            |             ISIC              |        Crop<br>Disease        |            EuroSAT            |             ISIC              |
|          Baseline          | **74.9 <br>$\pm$<br> 0.63%**  |   88.82 <br>$\pm$<br> 0.54%   |   78.23 <br>$\pm$<br> 0.61%   | **49.15 <br>$\pm$<br> 0.59%** |   86.90 <br>$\pm$<br> 0.56%   |   79.58 <br>$\pm$<br> 0.63%   | **48.03 <br>$\pm$<br> 0.63%** |
|          ProtoNet          |   63.22 <br>$\pm$<br> 0.71%   |   85.30 <br>$\pm$<br> 0.58%   |   75.93 <br>$\pm$<br> 0.69%   |   41.58 <br>$\pm$<br> 0.58%   |   84.02 <br>$\pm$<br> 0.61%   |   78.03 <br>$\pm$<br> 0.68%   |   42.88 <br>$\pm$<br> 0.56%   |
|       DSN<br>(dim=5)       |   70.09 <br>$\pm$<br> 0.69%   |   87.01 <br>$\pm$<br> 0.53%   |   76.52 <br>$\pm$<br> 0.71%   |   41.10 <br>$\pm$<br> 0.57%   | **89.21 <br>$\pm$<br> 0.52%** | **82.11 <br>$\pm$<br> 0.58%** | **43.84 <br>$\pm$<br> 0.58%** |
|       DSN<br>(dim=4)       | **71.54 <br>$\pm$<br> 0.65%** | **89.03 <br>$\pm$<br> 0.53%** | **78.28 <br>$\pm$<br> 0.66%** | **42.65 <br>$\pm$<br> 0.57%** |   86.24 <br>$\pm$<br> 0.58%   |   80.24 <br>$\pm$<br> 0.62%   |   41.36 <br>$\pm$<br> 0.58%   |
| DSN<br>(dim=4)<br>(sparse) |   71.31 <br>$\pm$<br> 0.65%   |  88.19% <br>$\pm$<br> 0.52%   |   78.13 <br>$\pm$<br> 0.66%   |   41.74 <br>$\pm$<br> 0.54%   | **87.22 <br>$\pm$<br> 0.53%** | **80.51 <br>$\pm$<br> 0.64%** | **42.94 <br>$\pm$<br> 0.55%** |

* If the highest performance is of baseline method, then the best performance of other methods are also marked.



<!-- ##### History

subspace: 64.82% +- 0.70%
* 100 epoch
* n_dim: 19
* learning rate: 0.001
* optimizer: AdamW
* weight_decay: default of AdamW (0.01)
* lambda: 0.03 (scale of discriminative loss, same as paper)

subspace: 69.59% +- 0.66%
* 200 epoch
    * 100 epoch (lr=0.001) + 100 epoch (lr=0.0001)
* n_dim: 19
* learning rate: 0.001 -> 0.0001
* optimizer: AdamW
* weight_decay: default of AdamW (0.01)
* lambda: 0.03

subspace: 71.28% +- 0.65%, 88.27% +- 0.50%, 77.24% +- 0.68%, 41.81% +- 0.56%
* 300 epoch
    * 100 epoch (lr=0.001) + 100 epoch (lr=0.0001) + 100 epoch (lr=0.00001)
* n_dim: 19
* learning rate: 0.001 -> 0.0001 -> 0.00001
* optimizer: AdamW
* weight_decay: default of AdamW (0.01)
* lambda: 0.03

subspace:
* FSL: 70.09% +- 0.69%
* CDFSL-Single: 87.01% +- 0.53%, 76.52% +- 0.71%, 41.10% +- 0.57%
* 300 epoch
    * 100 epoch (lr=0.001) + 100 epoch (lr=0.0001) + 100 epoch (lr=0.00001)
* n_dim: **5**
* learning rate: 0.001 -> 0.0001 -> 0.00001
* optimizer: AdamW
* weight_decay: default of AdamW (0.01)
* lambda: 0.03

subspace(multi):
* CDFSL-Multi: 89.21% +- 0.52%, 82.11% +- 0.58%, 43.84% +- 0.58%
    * domain 0 train acc (best): 69%
    * domain 1 train acc (best): 75%
* 300*2 epoch
* n_dim: **5**
* learning rate: 0.001 -> 0.0001 -> 0.00001
* optimizer: AdamW
* weight_decay: default of AdamW (0.01)
* lambda: 0.03

subspace (dim=4):
* train:
    * Single: last 71.31% +- 1.64%, best 72.42% +- 1.55%
* test:
    * FSL: 71.54% +- 0.65%
    * CDFSL-Single: 89.03% +- 0.53%, 78.28% +- 0.66%, 42.65% +- 0.57%
    * CDFSL-Multi: 86.24% +- 0.58%, 80.24% +- 0.62%, 41.36% +- 0.58%

subspace_plus (dim=4):
* train:
    * Single: last 69.01% +- 1.58% ,best 71.96% +- 1.57%
* test:
    * FSL: 71.31% +- 0.65%
    * CDFSL-Single: 88.19% +- 0.52%, 78.13% +- 0.66%, 41.74% +- 0.54%
    * CDFSL-Multi: 87.22% +- 0.53%, 80.51% +- 0.64% -->

### References

* https://github.com/chrysts/dsn_fewshot/
* https://github.com/JiaFong/NTU_aMMAI21_cdfsl
