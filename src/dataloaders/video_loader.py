import torch
import cv2


class VideoDataloader(torch.utils.data.Dataset):
    def __init__(self, video_path) -> None:
        super().__init__()
        self.vide_descriptor = cv2.VideoCapture(video_path)

    def __len__(self):
        return int(self.vide_descriptor.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        ok, frame = self.vide_descriptor.read()
        return frame
