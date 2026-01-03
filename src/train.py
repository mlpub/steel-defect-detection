# train model script

# import libraries
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, models

import config
import common_utils
import data_utils
import train_utils
import plot_utils



EXP_CODE = "E1-FINAL"
MODEL_SAVE_PATH = f"stage1-{EXP_CODE}-model.pth"
ONNX_SAVE_PATH = f"stage1-{EXP_CODE}-model.onnx"
ONNX_INT_SAVE_PATH = f"stage1-{EXP_CODE}-model-int8.onnx"

# override some config parameters for final training
config.IMAGE_SIZE = 384
config.EPOCHS = 20
config.LEARNING_RATE = 1e-5

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# set seed
common_utils.set_seed(config.SEED)

# load train csv data
df = pd.read_csv(config.TRAIN_CSV)
# split data into train, val, test (70/15/15)
train_df, val_df, test_df, _, _ = data_utils.split_data(df, config.SEED)
# combine train and val for final training
train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)
print(f"Train+Val size: {len(train_val_df)}")

# create transforms
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),   
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_val_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# create datasets
print("Creating datasets...")
train_dataset = data_utils.SteelDataset(df=train_val_df, 
                                    data_folder=config.IMAGE_FOLDER,
                                    transforms=train_transform,
                                    image_size=config.IMAGE_SIZE
                                    )
val_dataset = data_utils.SteelDataset(df=test_df, 
                                    data_folder=config.IMAGE_FOLDER,
                                    transforms=test_val_transform,
                                    image_size=config.IMAGE_SIZE
                                   )

# create data loaders
print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=config.PREFETCH_FACTOR
                          )

test_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=False,
                         num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=config.PREFETCH_FACTOR
                         )


# create model
class SteelModel(nn.Module):
    def __init__(self):
        super(SteelModel, self).__init__()

        self.base_model = models.mobilenet_v2(weights='IMAGENET1K_V1').features
        for param in self.base_model.parameters():
            param.requires_grad = False

        for p in self.base_model[-4:].parameters():
            p.requires_grad = True

        self.globalavg = nn.AdaptiveAvgPool2d((2, 2))

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(1280 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.base_model(x)
        #print(x.shape)
        x = self.globalavg(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        return x 


# initialize model, criterion, optimizer
model = SteelModel()
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# training loop
print("Starting training...")
for epoch in range(config.EPOCHS):
    train_utils.train_one_epoch(model, train_loader, criterion, optimizer, device)

    print(f"Epoch {epoch+1}/{config.EPOCHS}")


# save model weights
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# export to ONNX
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
torch.onnx.export(
    model,              # torch model in eval mode
    dummy_input,               # sample input tensor
    ONNX_SAVE_PATH,              # onnx output filename
    input_names=["input"],     # name of onnx graph input
    output_names=["output"],   # name of onnx graph output
    opset_version=18,          # onnx operator standard version, current latest value is 18
    do_constant_folding=True,  # precomputes constant operations to reduce size & improve speed
    dynamo=False,               # Use new exporter, need onnxscript (pip install onnxscript)
)






