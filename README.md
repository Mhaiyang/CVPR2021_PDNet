# CVPR2021_PDNet

## Depth-Aware Mirror Segmentation
[Haiyang Mei](https://mhaiyang.github.io/), [Bo Dong](https://dongshuhao.github.io/), Wen Dong, [Pieter Peers](http://www.cs.wm.edu/~ppeers/), [Xin Yang](https://xinyangdut.github.io/), Qiang Zhang, Xiaopeng Wei

[[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Depth-Aware_Mirror_Segmentation_CVPR_2021_paper.pdf)] [[Project Page](https://mhaiyang.github.io/CVPR2021_PDNet/index.html)]

### Abstract
We present a novel mirror segmentation method that leverages depth estimates from ToF-based cameras as an additional cue to disambiguate challenging cases where the contrast or relation in RGB colors between the mirror re-flection and the surrounding scene is subtle. A key observation is that ToF depth estimates do not report the true depth of the mirror surface, but instead return the total length ofthe reflected light paths, thereby creating obvious depth dis-continuities at the mirror boundaries. To exploit depth information in mirror segmentation, we first construct a large-scale RGB-D mirror segmentation dataset, which we subse-quently employ to train a novel depth-aware mirror segmentation framework. Our mirror segmentation framework first locates the mirrors based on color and depth discontinuities and correlations. Next, our model further refines the mirror boundaries through contextual contrast taking into accountboth color and depth information. We extensively validate our depth-aware mirror segmentation method and demonstrate that our model outperforms state-of-the-art RGB and RGB-D based methods for mirror segmentation. Experimental results also show that depth is a powerful cue for mirror segmentation.

### Citation
If you use this code, please cite:

```
@InProceedings{Mei_2021_CVPR,
    author    = {Mei, Haiyang and Dong, Bo and Dong, Wen and Peers, Pieter and Yang, Xin and Zhang, Qiang and Wei, Xiaopeng},
    title     = {Depth-Aware Mirror Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021}
}
```

### Requirements
* PyTorch == 1.0.0
* TorchVision == 0.2.1
* CUDA 10.0  cudnn 7.2

### Test
Download 'resnet50-19c8e357.pth' at [here](https://download.pytorch.org/models/resnet50-19c8e357.pth) and trained model 'PDNet.pth' at [here](https://mhaiyang.github.io/CVPR2021_PDNet/index.html), then run `infer.py`.

### License
Please see `license.txt`

### Contact
E-Mail: mhy666@mail.dlut.edu.cn