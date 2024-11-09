import unittest
from src.dataloaders.video_loader import VideoDataloader
from src.utils.constants import Config
import os
import numpy as np
import cv2
import shutil
import glob
import torchvision
from torch.utils.data import DataLoader


class DataloaderSuitCase(unittest.TestCase):
    def setUp(self):
        super(DataloaderSuitCase, self).setUp()
        self.config = Config(config_file_path="test_run_conf.yaml")
        self.loader = VideoDataloader(video_path=os.path.join(self.config.get_config["input"]["path"], "test.avi"),
                                transform=torchvision.transforms.Resize((640, 640),
                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                     max_size=None,
                                                                     antialias=True))
        self.test_images_output_path = os.path.join(self.config.get_config["output"]["path"], "test_images")
        shutil.rmtree(self.test_images_output_path, ignore_errors=True)
        os.makedirs(self.test_images_output_path, exist_ok=True)

    def test_get_5_different_frames(self):
        frames = []
        i = 0
        for frame in self.loader:
            frames.append(frame)
            i += 1
            if i == 5:
                break

        self.assertEqual(len(frames), 5)
        self.assertTrue(np.sum(np.abs(frames[0] - frames[-1])) > 0)

    def test_save_100_first_frames_to_disk(self):
        i = 0
        for index, f in enumerate(self.loader):
            cv2.imwrite(os.path.join(self.test_images_output_path, f"test_image_{index}.png"), f)
            i += 1
            if i == 100:
                break

        saved_images = os.listdir(self.test_images_output_path)
        saved_images.sort()

        self.assertEqual(saved_images[0], "test_image_0.png")
        self.assertEqual(len(glob.glob(os.path.join(self.test_images_output_path, "*.png"))), 100)

    def test_get_all_frames_from_video(self):
        frames = []
        i = 0
        while i < self.loader.__len__():
            frames.append(next(iter(self.loader)))
            i += 1

        self.assertEqual(len(frames), self.loader.__len__())
        self.assertTrue(np.sum(np.abs(frames[0] - frames[-1])) > 0)
