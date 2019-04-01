# Multiple Granularity Network
Reproduction of paper: Learning Discriminative Features with Multiple Granularities for Person Re-Identification

### About

This is a **non-official** pytorch re-production of paper: [Learning Discriminative Features with Multiple 
Granularities for Person Re-Identification](https://arxiv.org/abs/1804.01438). Still **Work In Progress**.

Please cite and refer to:

```text
@ARTICLE{2018arXiv180401438W,
    author = {{Wang}, G. and {Yuan}, Y. and {Chen}, X. and {Li}, J. and {Zhou}, X.},
    title = "{Learning Discriminative Features with Multiple Granularities for Person Re-Identification}",
    journal = {ArXiv e-prints},
    archivePrefix = "arXiv",
    eprint = {1804.01438},
    primaryClass = "cs.CV",
    keywords = {Computer Science - Computer Vision and Pattern Recognition},
    year = 2018,
    month = apr,
    adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180401438W},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Code is **only** tested against python 2.7 and pytorch 0.4.

### Implementation

![Multiple Granularity Network](/architecture.png)

* [mgn/mgn.py](/mgn/mgn.py): re-production of Multiple Granularity Network.

* [mgn/ide.py](/mgn/ide.py): baseline ResNet-50 based model, which is a rewritten from [Person reID baseline pytorch](
https://github.com/layumi/Person_reID_baseline_pytorch).

* [mgn/triplet.py](/mgn/triplet.py): triplet semi-hard sample mining loss.

* [mgn/market1501.py](/mgn/market1501.py): Market-1501 dataset.

* `Market-1501-v15.09.15/`: Market-1501 dataset root directory.

### Current Progress

* 2018-04-28: mAP=0.579464, r@1=0.798694, r@5=0.909739, r@10=0.938539
