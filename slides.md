#### (This is presentation slides)

## aMMAI 2021 Final Project
##### B07902055 資工三 謝宗晅

---



### Tasks
| Task         |        Training         |          Testing           |
| ------------ |:-----------------------:|:--------------------------:|
| FSL          |      mini-ImageNet      |       mini-ImageNet        |
| CDFSL-Single |      mini-ImageNet      | CropDisease, EuroSAT, ISIC |
| CDFSL-Multi  | mini-ImageNet, Cifar100 | CropDisease, EuroSAT, ISIC |

----

#### Brief Overview of Datasets

* [mini-ImageNet](https://deepai.org/dataset/imagenet)
    * Random 100 classes chosen from ImageNet
    * 600 images per class
* [CropDisease](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)
    * Images of plant leaves
    * Class: crop disease
* [EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat)
    * Photos taken by satellite
    * Class: land usage
* [ISIC](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/aboutIsicOverview)
    * Images of skin lesions
    * Class: skin diseases
* [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)
    * 100 classes with 600 images per class

---

## My Baseline

----

#### Deep Subspace Network (DSN)

----

#### Concept

* Classified by "subspace" of classes
* Measured by distances to subspaces of classes

----

#### Distance Measurement

$$d_c(\mathbf q) = \Big\| (\mathbf I - \mathbf M_c) (f_{\Theta}(\mathbf q) - \boldsymbol \mu_c) \Big\|^2$$

where

$$\begin{aligned}\mathbf M_c &= \mathbf P_c \mathbf P_c^T \\ \mathbf P_c &= \text{truncated basis of feature space}\end{aligned}$$

----

#### Distance Measurement (Contd.)

(Our task is 5-shot)
$$\mathbf {\tilde{X}_c} = [f_{\Theta}(\mathbf x_{c,1}) - \boldsymbol \mu_c, \cdots, f_{\Theta}(\mathbf x_{c,5}) - \boldsymbol\mu_c]$$

$$\mathbf {\tilde{X}_c} = \mathbf U \mathbf S \mathbf V^T \text{(by SVD)}$$

$$\mathbf P_c = \text{truncated } \mathbf U$$

----

#### Output

* Apply softmax function to output logits

$$p_{c,\mathbf q} = p(c \mid \mathbf q) = \frac{\exp(-d_c(\mathbf q))}{\sum_{c'} \exp(-d_{c'}(\mathbf q))}$$

----

#### Loss Function

$$\sum_{c}\log(p_{c,\mathbf q}) + \lambda \sum_{i \ne j}\Big\| \mathbf P_i \mathbf P_j^T \Big\|^2_F$$

(classification loss + discriminative loss)

----

#### Evaluation Results


|     Method     |              FSL              |         CDFSL-Single          |                               |                               |          CDFSL-Multi          |                               |                               |
|:--------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|                |       mini-<br>ImageNet       |        Crop<br>Disease        |            EuroSAT            |             ISIC              |        Crop<br>Disease        |            EuroSAT            |             ISIC              |
| DSN<br>(dim=5) |   70.09 <br>$\pm$<br> 0.69%   |   87.01 <br>$\pm$<br> 0.53%   |   76.52 <br>$\pm$<br> 0.71%   |   41.10 <br>$\pm$<br> 0.57%   | **89.21 <br>$\pm$<br> 0.52%** | **82.11 <br>$\pm$<br> 0.58%** | **43.84 <br>$\pm$<br> 0.58%** |
| DSN<br>(dim=4) | **71.54 <br>$\pm$<br> 0.65%** | **89.03 <br>$\pm$<br> 0.53%** | **78.28 <br>$\pm$<br> 0.66%** | **42.65 <br>$\pm$<br> 0.57%** |   86.24 <br>$\pm$<br> 0.58%   |   80.24 <br>$\pm$<br> 0.62%   |   41.36 <br>$\pm$<br> 0.58%   |

----

#### Discussion

$dim=n$ means the dimension of subspace

(the number of bases in $\mathbf P$)

The official setting of DSN is $dim=4$

($dim=N-1$ for $N$-way $K$-shot setting)

----

#### Discussion (Contd.)

Overall, the official setting is better than $dim=5$ setting.

But in **CDFSL-Multi** task, $dim=5$ is better than official setting.

I hypothesize that it is because the model of official setting has not converged yet.

---

## My Model

----

#### Intuition

* The discriminative loss only represents "inter-class" discriminative but no "intra-class" compactness.

* Therefore, I want to enable "intra-class" compactness based on DSN.

* Or, add another term of "inter-class" discriminative?

----

#### Initial Approach

Add loss based on singular values of $\mathbf{\tilde X}_c$:

* I pressumed if the explained variace are scattered in less dimensions, then the feature space would be "flatter".
* Let singular values be: $s_1, s_2, \cdots, s_5$
* The new loss function:
$$\mathcal L = \text{DSNLoss} + \lambda'\Big(-\frac{\sum_{i=1}^5 is_i}{\sum_{i=1}^5 s_i}\Big)$$

----

#### Ideally

|                   Before                   |                   After                    |
|:------------------------------------------:|:------------------------------------------:|
| ![](https://i.imgur.com/c7oqVbL.png) | ![](https://i.imgur.com/nlnGXDD.png) |


----

#### Result

* It is not trainable. (really bad performance)
(I tried many choices of hyperparameters and methods to enable the "flatness" of feature space.)
* It is like enable sparsity in another feature space. (of dimension 4 but not 512)

----

#### Approach 2

Add L1 regularizer on bases of subspace:

* If the basis is sparser, then the subspace is more likely to be discriminative (?)
    * If different subspace are constructed by different dimension of feature, will the subspace be more discriminative?
* Recall:
    $$d_c(\mathbf q) = \Big\| (\mathbf I - \mathbf P_c\mathbf P_c^T) (f_{\Theta}(\mathbf q) - \boldsymbol \mu_c) \Big\|^2$$
* The new loss function:
    $$\mathcal L = \text{DSNLoss} + \lambda' \sum_{i}\Big\| \mathbf P_{i}\Big\|_1$$


----


#### Result

|           Method           |              FSL              |         CDFSL-Single          |                               |                               |          CDFSL-Multi          |                               |                               |
|:--------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|                            |       mini-<br>ImageNet       |        Crop<br>Disease        |            EuroSAT            |             ISIC              |        Crop<br>Disease        |            EuroSAT            |             ISIC              |
|       DSN<br>(dim=4)       | **71.54 <br>$\pm$<br> 0.65%** | **89.03 <br>$\pm$<br> 0.53%** | **78.28 <br>$\pm$<br> 0.66%** | **42.65 <br>$\pm$<br> 0.57%** |   86.24 <br>$\pm$<br> 0.58%   |   80.24 <br>$\pm$<br> 0.62%   |   41.36 <br>$\pm$<br> 0.58%   |
| DSN<br>(dim=4)<br>(sparse) |   71.31 <br>$\pm$<br> 0.65%   |  88.19% <br>$\pm$<br> 0.52%   |   78.13 <br>$\pm$<br> 0.66%   |   41.74 <br>$\pm$<br> 0.54%   | **87.22 <br>$\pm$<br> 0.53%** | **80.51 <br>$\pm$<br> 0.64%** | **42.94 <br>$\pm$<br> 0.55%** |

----

#### Discussion

* It seems that the regularizer doesn't have large impact on performance.
    * Possible reasons:
        * The dimension of feature is not large enough to enable sparse feature.
        * The sparsity should be applied on feature, but not bases. (?)
        * The hyperparameter scale is not appropriate.
* Only on task "cdfsl-multi", the proposed method performs better than original DSN.
* We can see that applying L1 regularizer on the basis makes training on Cifar100 a little bit easier. (Training curve shows that)
    * Possible reason may be that Cifar100 images have less content so that model is able to learn a sparser feature.

---

### Other Details

----

#### Training Detail

* All models are trained for 300 epochs
    * Learning rate is initially 0.001, and multiplied by 0.1 every 100 epochs.
    * AdamW optimizer is used with default setting (except for learning rate).
* Hyperparameters
    * $\lambda_1=0.03$: Weight of discriminative term in loss function (identical to official setting of paper).
    * $\lambda_2=0.001$: Weight of sparsity term in loss function.

----

#### Training Curve

![](https://i.imgur.com/E7IXy2b.png)

----

#### Result Table (Full)

|  Method  |             FSL              |         CDFSL-Single          |                               |                               |          CDFSL-Multi          |                               |                               |
|:--------:|:----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
|          |      mini-<br>ImageNet       |        Crop<br>Disease        |            EuroSAT            |             ISIC              |        Crop<br>Disease        |            EuroSAT            |             ISIC              |
| Baseline | **74.9 <br>$\pm$<br> 0.63%** | **88.82 <br>$\pm$<br> 0.54%** | **78.23 <br>$\pm$<br> 0.61%** | **49.15 <br>$\pm$<br> 0.59%** | **86.90 <br>$\pm$<br> 0.56%** | **79.58 <br>$\pm$<br> 0.63%** | **48.03 <br>$\pm$<br> 0.63%** |
| ProtoNet |  63.22 <br>$\pm$<br> 0.71%   |   85.30 <br>$\pm$<br> 0.58%   |   75.93 <br>$\pm$<br> 0.69%   |   41.58 <br>$\pm$<br> 0.58%   |   84.02 <br>$\pm$<br> 0.61%   |   78.03 <br>$\pm$<br> 0.68%   |   42.88 <br>$\pm$<br> 0.56%   |
<br>

* Baseline method performs the best on FSL and all ISIC tasks.

----

#### Result Table (Full) (Contd.)

|           Method           |              FSL              |         CDFSL-Single          |                               |                               |           CDFSL-Multi           |                                 |                               |
|:--------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-------------------------------:|:-------------------------------:|:-----------------------------:|
|       DSN<br>(dim=5)       |   70.09 <br>$\pm$<br> 0.69%   |   87.01 <br>$\pm$<br> 0.53%   |   76.52 <br>$\pm$<br> 0.71%   |   41.10 <br>$\pm$<br> 0.57%   | ***89.21 <br>$\pm$<br> 0.52%*** | ***82.11 <br>$\pm$<br> 0.58%*** | **43.84 <br>$\pm$<br> 0.58%** |
|       DSN<br>(dim=4)       | **71.54 <br>$\pm$<br> 0.65%** | **89.03 <br>$\pm$<br> 0.53%** | **78.28 <br>$\pm$<br> 0.66%** | **42.65 <br>$\pm$<br> 0.57%** |    86.24 <br>$\pm$<br> 0.58%    |    80.24 <br>$\pm$<br> 0.62%    |   41.36 <br>$\pm$<br> 0.58%   |
| DSN<br>(dim=4)<br>(sparse) |   71.31 <br>$\pm$<br> 0.65%   |  88.19% <br>$\pm$<br> 0.52%   |   78.13 <br>$\pm$<br> 0.66%   |   41.74 <br>$\pm$<br> 0.54%   |  **87.22 <br>$\pm$<br> 0.53%**  |  **80.51 <br>$\pm$<br> 0.64%**  | **42.94 <br>$\pm$<br> 0.55%** |
<br>

* My proposed method performs better than oracle on CDFSL-Multi
* For DSN (dim=5) method, the training process is not identical with other DSNs. Thus, I marked the best performance among other DSNs.

----

#### Conclusion

* Applying L1 regularizer on bases of subspaces can help a little bit on training with Cifar100.
    * However the performance after convergence is unknown.
* Using singular value in loss function seems infeasible.
* It takes at least 200 epochs for model to converge on a dataset. (Under my training setting)

----

#### Future Work

* Further investigate on the effictiveness of L1 regularizer with math proof.
* Try other backbone networks (ResNet18, ResNet34)


---

### References

* [NTU-AMMAI-CDFSL (TA's repo)](https://github.com/JiaFong/NTU_aMMAI21_cdfsl)
* [Adaptive Subspaces for Few-Shot Learning (Repo of paper)](https://github.com/chrysts/dsn_fewshot)
* [mini-ImageNet](https://deepai.org/dataset/imagenet)
* [CropDisease](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)
* [EuroSAT](https://www.tensorflow.org/datasets/catalog/eurosat)
* [ISIC](https://www.isic-archive.com/#!/topWithHeader/tightContentTop/about/aboutIsicOverview)
* [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)

---

#### Q&A

----

### Thanks For Your Time
