import cv2
import torchvision
import torch
from decord import VideoReader
from decord import cpu

# class VideoDataloaderDecord(torch.utils.data.Dataset):
#     def __init__(self, video_path, transform=None):
#         self.video_path = video_path
#         self.transform = transform
#         self.total_frames = len(VideoReader(video_path, ctx=cpu(0)))
#
#     def __len__(self):
#         return self.total_frames
#
#     def __getitem__(self, idx):
#         reader = VideoReader(self.video_path, ctx=cpu(0))  # Initialize in each worker
#         frame = reader[idx].asnumpy()
#         if self.transform:
#             frame = self.transform(frame)
#         return frame


class MyIterable:
    def __init__(self, items):
        self.items = items
        self.index = 0  # Tracks the current position

    def __iter__(self):
        # The __iter__ method should return the iterator object itself.
        return self

    def __next__(self):
        # __next__ should return the next item or raise StopIteration when done.
        if self.index < len(self.items):
            item = self.items[self.index]
            self.index += 1
            return item
        else:
            raise StopIteration  # No more items to produce



class VideoFramesGenerator:
    def __init__(self, video_path, transform=None, num_threads=1, batch_size=256):
        super().__init__()
        self.video_path = video_path
        self.transform = transform
        self.num_threads = num_threads
        self.batch_size = batch_size
        self.length = len(VideoReader(video_path, ctx=cpu(0), num_threads=num_threads))
        self.index = 0

    def __iter__(self):
        # The __iter__ method should return the iterator object itself.
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        reader = VideoReader(self.video_path, ctx=cpu(0), num_threads=self.num_threads)
        if self.index < self.__len__():
            frames = reader[self.index: self.index + self.batch_size]
            self.index += self.batch_size

            # Convert frames to numpy and apply the transformation
            if self.transform:
                batch = [self.transform(frame) for frame in frames.asnumpy()]
                return torch.stack(batch)

            # return frames.asnumpy() #(torch.tensor(frames.asnumpy())/255.).transpose(1, 3)
            return [cv2.resize(frame, (256, 256)) for frame in frames.asnumpy()]
        else:
            raise StopIteration


class VideoDataloaderDecord(torch.utils.data.Dataset):
    def __init__(self, video_path, transform=None):
        super().__init__()
        self.video_path = video_path
        self.transform = transform
        self.reader = VideoReader(video_path, ctx=cpu(0))
        self.total_frames = len(self.reader)

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        frame = self.reader[idx]# Decord provides frame-accurate seeking
        frame = frame.asnumpy()
        if self.transform:
            frame = self.transform(frame)
        return frame


class VideoDataloaderPytorch(torch.utils.data.Dataset):
    def __init__(self, video_path, transform=None):
        super().__init__()
        self.video_path = video_path
        self.transform = transform if transform is not None else torchvision.transforms.ToTensor()

        # Initialize video reader for frame count only
        self._init_video_reader()
        self.total_frames = self.reader.container.streams.video[0].frames
        self.reader = None  # Remove reader to avoid problems with DataLoader forks

    def _init_video_reader(self):
        # Initialize reader as a private method to be used where needed
        self.reader = torchvision.io.VideoReader(self.video_path, "video")

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        if self.reader is None:
            self._init_video_reader()  # Initialize reader if not already present

        # Seek to frame and retrieve it
        self.reader.seek(idx)
        frame = next(self.reader)

        # Apply transform and return frame data
        return self.transform(frame["data"])


# class VideoDataloaderPytorch(torch.utils.data.Dataset):
#     def __init__(self, video_path, transform) -> None:
#         super().__init__()
#         self.video_path = video_path
#         self.transform = transform
#         self.reader = torchvision.io.VideoReader(video_path, "video")
#
#     def __len__(self):
#         return self.reader.container.streams.video[0].frames
#
#     def __getitem__(self, idx):
#         self.reader.seek(idx)
#         frame = next(self.reader)
#
#         return self.transform(frame["data"])


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
