## Deep Ensembles for Probabilistic Object Detection 

### Overview
![model image](images/model.png)

Probabilistic object detection is the task of detecting objects in images and
accurately quantifying the spatial and semantic uncertainties of the detection.
Quantifying uncertainty is critical in real-world robotic applications.
Traditional detection models can be ambiguous even when they provide a
high-probability output. Robot actions based on high-confidence, yet unreliable
predictions, can result in serious consequences. 

This repository provides source code for our 2021 CASE paper entitled "[An
Uncertainty Estimation Framework for Probabilistic Object
Detection](https://arxiv.org/pdf/2106.15007.pdf)." Our framework employs deep
ensembles and Monte Carlo dropout for approximating predictive uncertainty, and
it improves upon the uncertainty estimation quality of the baseline method. We
evaluate our approach on publicly available synthetic image datasets captured
from sequences of video.

If you find this project useful, then please consider citing our work: 

Z. Lyu, N.B. Gutierrez and W.J. Beksi, "An Uncertainty Estimation Framework for
Probabilistic Object Detection," *IEEE International Conference on Automation
Science and Engineering (CASE)*, 2021.

```bibtex
@inproceedings{lyu2021uncertainty,
  title={An Uncertainty Estimation Framework for Probabilistic Object Detection},
  author={Lyu, Zongyao and Gutierrez, Nolan B and Beksi, William J},
  booktitle={Proceedings of the IEEE International Conference on Automation Science and Engineering (CASE)},
  pages={},
  year={2021}
}
```

### Installation
#### Prerequisites

- Linux, MacOS, or Windows machine
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

Our code is based on the open-source object detection toolbox
[MMDetection](https://github.com/open-mmlab/mmdetection). The version we used in
our work is 2.0. To use our code, you will need to install MMDetection first.
You can find more information about this process on the MMDetection
[installation
instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
page.

### Usage

It's easy to run our code after you successfully install MMDetection. In order
to make it more efficient, we separate the process of producing bounding boxes
(and their corresponding labels) and performing inference. To produce boxes and
labels, just set ``reuseFiles = False`` and ``saveFiles = True``. Then, run the
following script

```shell
  python tools/testPOD.py 
```

The boxes and labels file will be saved as a pickle file in the ``savedOutputs``
folder. To do inference with an existing boxes and labels file, simply set
``reuseFiles = True`` and ``saveFiles = False``. There are four primary
hyperparameters we tuned in our work.

- thresholds: Threshold for detection score
- covPercents: Percent in which the covariance is scaled
- boxratios: Ratio by which the bounding box is reduced
- iouthresholds: IoU threshold

Given each pair of produced boxes and labels, you can run inference using a
combination of different values of hyperparameters. After you set the values of
these parameters to the ones you want to test, just run the above script again.
The computed PDQ score will be displayed in the terminal after inference
finishes.

### License 

[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/robotic-vision-lab/Deep-Ensembles-For-Probabilistic-Object-Detection/blob/main/LICENSE)
