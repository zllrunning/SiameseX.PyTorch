# SiameseX.PyTorch
A simplified PyTorch implementation of Siamese networks for tracking: SiamFC, SiamVGG, SiamDW, SiamRPN. 

**Warning:  It is still in development, some files are just for the integrity of the project.**

## Dependencies
- python2.7
- pytorch == 0.4.0
- opencv
- mmcv

## Currently supported models
- [SiamFC](https://arxiv.org/abs/1606.09549)
- [SiamVGG](https://arxiv.org/abs/1902.02804)
- [SiamFCRes22](https://arxiv.org/abs/1901.01660)
- [SiamFCIncep22](https://arxiv.org/abs/1901.01660)
- [SiamFCNext22](https://arxiv.org/abs/1901.01660)
- [SiamRPN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)
- [SiamRPNVGG](https://github.com/leeyeehoo/SiamRPN-VGG)
- [SiamRPNRes22](https://arxiv.org/abs/1901.01660)
- [SiamRPNIncep22](https://arxiv.org/abs/1901.01660)
- [SiamRPNResNeXt22](https://arxiv.org/abs/1901.01660)


## Backbones
- AlexNet
- VGG
- ResNet22
- Incep22
- ResNeXt22

## Demo
- Clone this repo and run
```
python demo.py --model SiamFCNext22
```

You can change `--mdoel` to other models like
```
python demo.py --model SiamFC
```

- You'll see the following:
<div align="center">
  <img src="data/bag.gif" width="400px" />
</div>

## Training

- download [VID dataset](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php) (I did not use Youtube-bb because of limited resources).
- download label files from (here)[https://drive.google.com/open?id=13aC_2stCEU0VoiIpp6wgudenUSVye74b] and run `label_preprocess.py --file --output_file --path` to replace my file path by yours,  *these three arguments should be treated carefully*.
- train SiamFCNext22 tracker
```
python train.py --gpu 0 --config configs/SiamFC.py
```
- just replace `--config` argument and you can train other models
```
python train.py --gpu 0 --config configs/SiamRPN.py
```

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  - [ ] Add testing code on common datasets
  - [x] Add SiamRPN(AlexNet as backbone)
  - [x] Add SiamRPN(VGG as backbone)
  - [x] Add SiamRPN(ResNet, ResNext, Inception as backbone)

### Citation 

```
@inproceedings{bertinetto2016fully,
  title={Fully-convolutional siamese networks for object tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Joao F and Vedaldi, Andrea and Torr, Philip HS},
  booktitle={European conference on computer vision},
  pages={850--865},
  year={2016},
  organization={Springer}
}

@inproceedings{Li2019SiamVGGVT,
  title={SiamVGG: Visual Tracking using Deeper Siamese Networks},
  author={Yuhong Li and Xiaofan Zhang},
  year={2019}
}

@inproceedings{SiamDW_2019_CVPR,
    author={Zhang, Zhipeng and Peng, Houwen},
    title={Deeper and Wider Siamese Networks for Real-Time Visual Tracking},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2019}
}
```
## References
- [SiamVGG](https://github.com/leeyeehoo/SiamVGG)
- [siamfc-pytorch](https://github.com/huanglianghua/siamfc-pytorch)
- [SiamDW](https://github.com/researchmm/SiamDW)













