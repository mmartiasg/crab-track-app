import torch
import cv2


class VideoDataloader(torch.utils.data.Dataset):
    def __init__(self, video_path, transform) -> None:
        super().__init__()
        self.video_path = video_path
        self.transform = transform

    def __len__(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)
        # set frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        # read next
        ret, frame = cap.read()
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame)
