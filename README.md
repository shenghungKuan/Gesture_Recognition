# Gesture_Recognition

## Environment Set Up
Install PyTorch with CUDA
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install additional dependencies
```
pip3 install matplotlib pandas pillow torchtnt==0.2.0 tqdm
```

Install extra dependencies for pandas
```
pip3 install tabulate pyarrow fastparquet
```

Install package for creating visually distinct colormaps
```
pip3 install distinctipy
```

Install utility packages
```
pip3 install cjm_pandas_utils cjm_psl_utils cjm_pil_utils cjm_pytorch_utils cjm_yolox_pytorch cjm_torchvision_tfms
```

## Reference
- [train_guide](https://github.com/cj-mills/pytorch-yolox-object-detection-tutorial-code/blob/main/notebooks/pytorch-yolox-object-detector-training.ipynb) 
- [YOLO_X Pytorch implementation model_ref](https://github.com/Megvii-BaseDetection/YOLOX) 
- [YOLO_X Paper](https://arxiv.org/abs/2107.08430) 
- [YOLO X model_ref](https://github.com/MegEngine/YOLOX?tab=readme-ov-file)

## Cite
```
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```


