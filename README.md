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

## Data resource

```
HuggingFace Dataset:	cj-mills/hagrid-sample-30k-384p
Archive Path:	E:\Datasets\..\Archive\hagrid-sample-30k-384p.zip
Dataset Path:	E:\Datasets\hagrid-sample-30k-384p
```

url:https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip

## Different workplace

- **gesture_recognition.ipynb** should be execute on Mac (Marcus)
- **gesture_recognition_wilson.ipynb** should be execute on Windows (Wilson)

## Use the trained model

If you don't want to train the model again, which is time-comsuming, you can use this trained model instead

```
# declare a model first
model_type = 'yolox_tiny'
model = build_model(model_type, len(class_names), pretrained=true).to(device=device, dtype=dtype)

```

Now load our model

```
PATH = f'final_model/yolox_tiny.pth'
model_state_dict = torch.load(PATH, map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
# load_state_dict() function takes a dictionary object, NOT a path to a saved object
```

## CheckPoints

If you want to train model by your self, the default path is under the directory below

```
${your_project_name}/{timestamp}/
```

so you could easily load your trained model in those directory

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
