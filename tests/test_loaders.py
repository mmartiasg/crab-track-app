import unittest
from src.utils.constants import Config
import os
import numpy as np
import cv2
import shutil
import glob
import torchvision
from torch.utils.data import DataLoader
from src.dataloaders.video_loader import VideoDataloader, VideoDataloaderPytorch, VideoDataloaderDecord, \
    VideoFramesGenerator
import torch


class DataloaderSuitCase(unittest.TestCase):
    def setUp(self):
        super(DataloaderSuitCase, self).setUp()
        self.config = Config(config_file_path=os.path.join(os.path.dirname(__file__), "test_run_conf.yaml"))
        self.test_images_output_path = os.path.join(os.path.dirname(__file__), self.config.get_config["output"]["path"], "test_images")
        shutil.rmtree(self.test_images_output_path, ignore_errors=True)
        os.makedirs(self.test_images_output_path, exist_ok=True)

    def test_get_5_different_frames(self):
        video_frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        loader = VideoDataloader(video_path=os.path.join(os.path.dirname(__file__),
                                                         self.config.get_config["input"]["path"],
                                                         "test_sample_1_720p.mp4"),
                                 transform=video_frame_transform)
        frames = []
        i = 0
        for frame in loader:
            frames.append(frame)
            i += 1
            if i == 5:
                break

        self.assertEqual(len(frames), 5)
        self.assertTrue(np.sum(np.abs(frames[0].numpy() - frames[-1].numpy())) > 0)

    def test_save_100_first_frames_to_disk(self):
        video_frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            lambda frame: torch.tensor(np.array(frame), dtype=torch.uint8),
        ])

        loader = VideoDataloader(video_path=os.path.join(os.path.dirname(__file__),
                                                         self.config.get_config["input"]["path"],
                                                         "test_sample_1_720p.mp4"),
                                 transform=video_frame_transform)

        data_loader = DataLoader(loader, batch_size=100, shuffle=False, num_workers=0)

        for batch in data_loader:
            for index, frame in enumerate(batch):
                cv2.imwrite(os.path.join(os.path.dirname(__file__),
                                         self.test_images_output_path,
                                         f"test_image_{index}.png"),
                            frame.numpy()
                            )
            break

        saved_images = os.listdir(os.path.join(os.path.dirname(__file__), self.test_images_output_path))
        saved_images.sort()

        self.assertEqual(saved_images[0], "test_image_0.png")
        self.assertEqual(len(glob.glob(os.path.join(os.path.dirname(__file__), self.test_images_output_path, "*.png"))), 100)

    def test_get_all_frames_from_video_with_open_cv_backend_data_loader_returns_batch_with_1024_samples(self):
        video_frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        loader = VideoDataloader(video_path=os.path.join(os.path.dirname(__file__),
                                                         self.config.get_config["input"]["path"],
                                                         "test_sample_2_720p.mp4"),
                                 transform=video_frame_transform)

        data_loader = DataLoader(loader, batch_size=1024, shuffle=False, num_workers=0)

        batches = []
        frames_count = 0
        for batch in data_loader:
            batches.append(batch)
            frames_count += len(batch)

        self.assertEqual(batches[0].shape, (1024, 3, 128, 128))
        self.assertEqual(batches[0][0].shape, (3, 128, 128))
        self.assertEqual(frames_count, loader.__len__())
        self.assertTrue(np.sum(np.abs(batches[0][0].numpy() - batches[-1][0].numpy())) > 0)

    def test_get_all_frames_from_video_with_pytorch_video_reader_backend_data_loader_returns_batch_with_1024_samples(
            self):
        video_frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])
        loader = VideoDataloaderPytorch(video_path=os.path.join(os.path.dirname(__file__),
                                                                self.config.get_config["input"]["path"],
                                                                "test_sample_2_720p.mp4"),
                                        transform=video_frame_transform)

        data_loader = DataLoader(loader, batch_size=1024, shuffle=False, num_workers=0)

        batches = []
        frames_count = 0
        for batch in data_loader:
            batches.append(batch)
            frames_count += len(batch)

        self.assertEqual(batches[0].shape, (1024, 3, 128, 128))
        self.assertEqual(batches[0][0].shape, (3, 128, 128))
        self.assertEqual(frames_count, loader.__len__())
        self.assertTrue(np.sum(np.abs(batches[0][0].numpy() - batches[-1][0].numpy())) > 0)

    def test_get_all_frames_from_video_with_decord_backend_data_loader_returns_batch_with_256_samples(self):
        video_frame_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])

        batch_size = 256

        loader = VideoFramesGenerator(video_path=os.path.join(os.path.dirname(__file__),
                                                              self.config.get_config["input"]["path"],
                                                              "test_sample_2_720p.mp4"),
                                      transform=video_frame_transform,
                                      batch_size=batch_size)

        batches = []
        frames_count = 0
        for batch in loader:
            batches.append(batch)
            frames_count += len(batch)

        self.assertEqual(len(batches[0]), batch_size)
        self.assertEqual(batches[0][0].shape, (3, 128, 128))
        self.assertEqual(loader.__len__(), frames_count)
        self.assertTrue(np.sum(np.abs(batches[0][0].numpy() - batches[-1][0].numpy())) > 0)
