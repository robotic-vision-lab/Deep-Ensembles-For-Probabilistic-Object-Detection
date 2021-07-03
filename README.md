## Deep Ensembles For Probabilistic Object Detection 

### Overview
![model image](images/model.png)

Probabilistic object detection is the task of detecting objects in images and
accurately quantifying the spatial and semantic uncertainties of the detection.
Quantifying uncertainty is critical in real-world robotic applications.
Traditional detection models can be ambiguous even when they provide a
high-probability output. Robot actions based on high-confidence, yet unreliable
predictions, can result in serious consequences. 

This repository provides source code for our 2021 CASE paper entitled 
"[An Uncertainty Estimation Framework for Probabilistic Object Detection](https://arxiv.org/pdf/2106.15007.pdf)." 
Our framework employs deep ensembles and Monte Carlo dropout for approximating
predictive uncertainty, and it improves upon the uncertainty estimation quality
of the baseline method. We evaluate our approach on publicly available
synthetic image datasets captured from sequences of video.

If you find this project useful, then please consider citing our work:
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

- Linux or macOS or Windows
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

Our code is based on the open source object detection toolbox [MMdetection](https://github.com/open-mmlab/mmdetection).
The version we used in our work is 2.0. To use our code, you need to install
mmdetection first. You can find more information about how to install mmdetection
in their [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md)
instruction page.

### Usage

It's easy to run our code after you successfully install mmdetection. 

In order to make it more efficient, we separate the process of producing bounding
boxes (and their corresponding labels) and doing inference. To produce boxes and
labels, just set reuseFiles = False, saveFiles = True. Then run the script below,
#### Test PrOD with pretrained model

   ```shell
   python tools/testPOD.py 
   ```
The boxes and labels file will be saved in savedOutputs folder.

Then to do inference with existing boxes and labels file, simply set reuseFiles = True,
saveFiles = False, and run the above script again. The result can be seen in the 
terminal after finishing the inference.

### License 

[Apache 2.0](https://github.com/robotic-vision-lab/Deep-Ensembles-For-Probabilistic-Object-Detection/blob/main/LICENSE)
