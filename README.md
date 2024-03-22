# CVPR24_OC_SCMNet

### One-Class Face Anti-spoofing via Spoof Cue Map-Guided Feature Learning (CVPR '24)

## Illustration of the proposed idea of SCM-guided feature learning for one-class face anti-spoofing (FAS).
![plot](figures/idea.pdf)

## Architecture of OC SCMNet
![plot](figures/framework_small.png)


## Training & Testing
Run `train.py` to train LDCformer

Run `test.py` to test LDCformer


## Citation

If you use the LDCformer/Decoupled-LDC, please cite the paper:
 ```
@inproceedings{huang2023ldcformer,
  title={LDCformer: Incorporating Learnable Descriptive Convolution to Vision Transformer for Face Anti-Spoofing},
  author={Huang, Pei-Kai and Chiang, Cheng-Hsuan and Chong, Jun-Xiong and Chen, Tzu-Hsien and Ni, Hui-Yu and Hsu, Chiou-Ting},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={121--125},
  year={2023},
  organization={IEEE}
}
 @inproceedings{huang2022learnable,
  title={Learnable Descriptive Convolutional Network for Face Anti-Spoofing},
  author={Huang, Pei-Kai and H.Y. Ni and Y.Q. Ni and C.T. Hsu},
  booktitle={BMVC},
  year={2022}
}
```
