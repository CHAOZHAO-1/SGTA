# SGTA ![GitHub visitors](https://visitor-badge.laobi.icu/badge?page_id=CHAOZHAO-1.SGTA&color=blue&style=flat-square)

## Paper

[EAAI 2026] Adaptive generalization: An active perception framework for robust intelligent fault diagnosis under dynamic environments

Paper link: [Adaptive generalization: An active perception framework for robust intelligent fault diagnosis under dynamic environments](https://www.sciencedirect.com/science/article/pii/S095219762602587X?dgcid=author)

## Abstract
The distribution shift problem induced by dynamic environments significantly degrades the performance of data- driven fault diagnosis systems in real-world applications. Although domain generalization techniques have been introduced to address distribution discrepancies, static generalized models passively handle all unseen domains and neglect the value of the testing data. To bridge this gap, this study investigates a novel learning paradigm termed adaptive generalization-based fault diagnosis and develops an active perception framework designed to improve robustness under dynamic operating conditions. The developed framework initially fosters universal generalization during offline training and subsequently, during deployment, actively perceives environmental information from the incoming inference data, facilitating targeted adaptation. Perturbation effect minimization is applied to the available multi-source domain data to achieve flat minima, providing a reliable initialization for subsequent online adaptation. The model parameters are then adjusted based on knowledge transferability and structural information extracted from the stream of unseen target domain data, allowing the model to effectively adapt to the inference environment. The effectiveness and superiority of the proposed framework are validated through extensive experiments on three failure datasets. 

##  Proposed Learning Paradigms 

![image](https://github.com/CHAOZHAO-1/MUGTN/blob/main/IMG1/F1.png)

##  Proposed Network 


![image](https://github.com/CHAOZHAO-1/MUGTN/blob/main/IMG1/F2.png)

##  BibTex Citation


If you like our paper or code, please use the following BibTex:

```
@article{ZHAO2026116303,
title = {Adaptive generalization: An active perception framework for robust intelligent fault diagnosis under dynamic environments},
journal = {Engineering Applications of Artificial Intelligence},
volume = {184},
pages = {116303},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.116303},
url = {https://www.sciencedirect.com/science/article/pii/S095219762602587X},
author = {Chao Zhao and Weiming Shen and Enrico Zio and Hui Ma},
keywords = {Intelligent fault diagnosis, Domain shift, Domain generalization, Test-time adaptation, Dynamic environment},
abstract = {The distribution shift problem induced by dynamic environments significantly degrades the performance of data-driven fault diagnosis systems in real-world applications. Although domain generalization techniques have been introduced to address distribution discrepancies, static generalized models passively handle all unseen domains and neglect the value of the testing data. To bridge this gap, this study investigates a novel learning paradigm termed adaptive generalization-based fault diagnosis and develops an active perception framework designed to improve robustness under dynamic operating conditions. The developed framework initially fosters universal generalization during offline training and subsequently, during deployment, actively perceives environmental information from the incoming inference data, facilitating targeted adaptation. Perturbation effect minimization is applied to the available multi-source domain data to achieve flat minima, providing a reliable initialization for subsequent online adaptation. The model parameters are then adjusted based on knowledge transferability and structural information extracted from the stream of unseen target domain data, allowing the model to effectively adapt to the inference environment. The effectiveness and superiority of the proposed framework are validated through extensive experiments on three failure datasets. Our code is publicly available at: https://github.com/CHAOZHAO-1/SGTA.}
}
```
