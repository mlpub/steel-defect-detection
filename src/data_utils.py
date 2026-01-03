import os
import re

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import config


class SteelDataset:
    def __init__(self, df, data_folder, transforms=None, 
                 image_size=config.IMAGE_SIZE, return_tensor=True,
                 stack_input=False):
        self.filepaths = df['ImageId'].values
        self.labels = df['IsDefect'].values
        self.data_folder = os.path.expanduser(data_folder)
        self.image_size = image_size
        self.transforms = transforms
        self.return_tensor = return_tensor
        self.stack_input = stack_input

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.filepaths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.stack_input:
            # image input is 1600x256, split it become 2 x (800x256)
            # the stack: (800x256) + (800x288) + (800x256)
            h, w, c = img.shape

            # Left strip: 800x256
            left_strip = img[:, 0:800, :]
            # Right strip: 800x256
            right_strip = img[:, 800:1600, :]

            # Grayscale avg strip: 800x288
            avg_color = np.mean(img, axis=(0, 1)).astype(np.uint8)
            gray_val = int(np.mean(avg_color))
            gray_strip = np.ones((288, 800, 3), dtype=np.uint8) * gray_val
            # Add small Gaussian noise
            noise = np.random.normal(0, 2, gray_strip.shape).astype(np.int16)
            gray_strip = np.clip(gray_strip.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Stack vertically: (left, gray, right)
            stacked_img = np.vstack([left_strip, gray_strip, right_strip])
            img = stacked_img

        label = self.labels[idx]
        if self.return_tensor:
            img = Image.fromarray(img)  # Convert to PIL Image for torchvision transforms
            img = self.transforms(img)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
        label = torch.tensor(label, dtype=torch.float32) if self.return_tensor else label
        return img, label


def split_data(
     df, 
     seed, 
     train_pct=0.7, 
     small_train_size=None, 
     small_val_size=None, 
     stratify_col='IsDefect'
):
    # split data into train, val, test 70/15/15
    test_size = 1 - train_pct
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, 
                                        stratify=df[stratify_col], shuffle=True)
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=seed, 
                                    stratify=test_df[stratify_col], shuffle=True)
    
    # small train/val set for quick experiments
    if small_train_size is None:
        train_small_df = None
    else:
        train_small_df, temp_small_df = train_test_split(train_df, train_size=small_train_size, random_state=seed, 
                                    stratify=train_df[stratify_col], shuffle=True)
        
    if small_val_size is None:
        val_small_df = None
    else:
        val_small_df, _ = train_test_split(temp_small_df, train_size=small_val_size, random_state=seed,
                                        stratify=temp_small_df[stratify_col], shuffle=True)

    return train_df, val_df, test_df, train_small_df, val_small_df

# def split_data(
#     df, 
#     seed, 
#     train_pct=0.7, 
#     val_pct=0.15, 
#     test_pct=0.15, 
#     small_train_size=None, 
#     small_val_size=None, 
#     stratify_col='IsDefect'
# ):
#     assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-6, "Percentages must sum to 1.0"
#     stratify_vals = df[stratify_col] if stratify_col is not None else None
#     train_df, temp_df = train_test_split(
#         df, test_size=(1 - train_pct), random_state=seed, stratify=stratify_vals, shuffle=True
#     )
#
#     val_rel = val_pct / (val_pct + test_pct)
#     stratify_temp = temp_df[stratify_col] if stratify_col is not None else None
#     val_df, test_df = train_test_split(
#         temp_df, test_size=(1 - val_rel), random_state=seed, stratify=stratify_temp, shuffle=True
#     )
#
#     if small_train_size is not None:
#         stratify_train = train_df[stratify_col] if stratify_col is not None else None
#         train_small_df, _ = train_test_split(
#             train_df, train_size=small_train_size, random_state=seed, stratify=stratify_train, shuffle=True
#         )
#     else:
#         train_small_df = train_df
#
#     if small_val_size is not None:
#         stratify_val = val_df[stratify_col] if stratify_col is not None else None
#         val_small_df, _ = train_test_split(
#             val_df, train_size=small_val_size, random_state=seed, stratify=stratify_val, shuffle=True
#         )
#     else:
#         val_small_df = val_df
#
#     return train_df, val_df, test_df, train_small_df, val_small_df
