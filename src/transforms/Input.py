import decord.ndarray
import torchvision
import torch
import cv2
import numpy as np


class NDArrayToTensor:
    def __init__(self, size: tuple):
        self.size = size
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(size),
            torchvision.transforms.ToTensor()
        ])

    def __call__(self, frames: decord.ndarray.NDArray) -> torch.Tensor:
        batch = [self.transform(frame) for frame in frames.asnumpy()]
        return torch.stack(batch)


class NDArrayToImage:
    def __init__(self, size: tuple):
        self.size = size

    def __call__(self, frames: decord.ndarray.NDArray) -> list[np.ndarray]:
        return [cv2.resize(frame, self.size) for frame in frames.asnumpy()]
