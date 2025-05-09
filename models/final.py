# Author: Charles R. Clark
# CS 6440 Spring 2024

import torch
import torch.nn as nn

class FinalModel(nn.Module):
    def __init__(self, init_img_size: int):
        super().__init__()

        # ================================================================
        # Implement final model architecture...
        # ================================================================
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )

        new_img_size = int(((init_img_size - 2 + (2 * 0)) / 2) + 1)
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=36, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm2d(num_features=36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=0, stride=2)
        )
        self.flatten = nn.Flatten(start_dim=1)

        new_img_size = int(((new_img_size - 2 + (2 * 0)) / 2) + 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=(36 * new_img_size * new_img_size), out_features=int((36 * new_img_size * new_img_size) / 6)),
            nn.BatchNorm1d(num_features=int((36 * new_img_size * new_img_size) / 6)),
            nn.ReLU(),
            nn.Linear(in_features=int((36 * new_img_size * new_img_size) / 6), out_features=3)
        )

        # ----------------------------------------------------------------
    
    def forward(self, X: torch.Tensor):
        out = None

        # ================================================================
        # Implement final model forward pass...
        # ================================================================
        out = self.conv_layer_1(X)
        out = self.conv_layer_2(out)
        out = self.flatten(out)
        out = self.fc_layers(out)

        # ----------------------------------------------------------------

        return out