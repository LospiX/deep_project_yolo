import os
from  ultralytics import YOLO
import torch
from IPython.display import Image  # for displaying images
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from datasets import load_dataset

############
#| PARAMS
#|
TRAINING_NAME = "small-tune-0"
MODEL_NAME = "yolov8s.pt"
EPOCHS = 250
#######


############
#| LOAD MODEL & DATASET
#|
device = torch.device("cuda:0")
print("Device: ", device)
model = YOLO(MODEL_NAME).to(device)
# Load the dataset with streaming
dataset = load_dataset('kili-technology/plastic_in_river', split='train', streaming=True)
#######


############
#| TRAIN
#|
results = model.train(
    data='./model_copy.yaml',
    name        = TRAINING_NAME,
    # freeze    = 4, #7
    epochs      = EPOCHS, #120, #60, #30
    # batch     =-1,
    seed        = 42, #0,
    cache     = True, #SUPERATO LIMITE SU COLAB
    workers     = 24,
    device      = device,
    # single_cls  = True,	#Default: False ->	Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.
    dropout     = 0.015, #Default: 0.0
    imgsz       = 800,
    momentum    = 0.98,
    lr0         = 0.0001,
    save        = True,
    save_period = 25,
    plots       = True,

    ### AUGMENTATION PARAMS
    hsv_v       = 0.7,
    shear       = 55,
    perspective = 0.0005,
    hsv_h       = 0.1,
    degrees     = 120
)