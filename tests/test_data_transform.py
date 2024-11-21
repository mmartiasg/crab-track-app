import unittest

import decord.ndarray

from src.transforms.Adapter import YoloAdapter
from ultralytics.engine.results import Results
import numpy as np
import torch
from src.transforms.Input import NDArrayToTensor, NDArrayToImage


class DataTransform(unittest.TestCase):

    def test_transform_batch_with_3_results_from_yolov8_last_one_is_none_first_and_second_returns_0237_0133_0165_01302_frame_index_equals_3_inference_time_in_batch_16217_per_sample_5405ms(self):
        yolo_adapter_data_transform = YoloAdapter(video_name="test.mp4", tracker_name="mock", dataset="test")

        bbox1 = [256, 256, 115, 250, 0, 0.9]
        bbox2 = [256, 256, 115, 250, 0, 0.9]

        results1 = Results(boxes=torch.tensor([bbox1]),
                          path=".", names={0: 'Crab', 1: 'Backgroud'},
                          orig_img=np.zeros((1920, 1080)))
        results1.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                          'preprocess': 0.00035390257835388184}

        results2 = Results(boxes=torch.tensor([bbox2]),
                          path=".", names={0: 'Crab', 1: 'Backgroud'},
                          orig_img=np.zeros((1920, 1080)))
        results2.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                                  'preprocess': 0.00035390257835388184}

        results3 = Results(boxes=torch.tensor([0, 0, 0, 0, 0, 0]),
                          path=".", names={0: 'Crab', 1: 'Backgroud'},
                          orig_img=np.zeros((1920, 1080)))
        results3.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                                  'preprocess': 0.00035390257835388184}

        results3.boxes = []

        json_response_batch = yolo_adapter_data_transform([results1, results2, results3])

        self.assertEqual(len(json_response_batch), 3)

        # just the first box
        self.assertAlmostEqual(round(json_response_batch[0][0]["x1"], 4), round(256/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["y1"], 4), round(256/1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["x2"], 4), round(115/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["y2"], 4), round(250/1920, 4), delta=1e-7)

        self.assertAlmostEqual(round(json_response_batch[1][0]["x1"], 4), round(256/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["y1"], 4), round(256/1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["x2"], 4), round(115/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["y2"], 4), round(250/1920, 4), delta=1e-7)

        self.assertAlmostEqual(json_response_batch[2][0]["x1"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["y1"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["x2"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["y2"], None)

        self.assertAlmostEqual(yolo_adapter_data_transform.inference_time_batch, 5.405731499195099 * 3, delta=1e-8)
        self.assertAlmostEqual(yolo_adapter_data_transform.inference_time_batch / 3,
                               yolo_adapter_data_transform.inference_time_sample, delta=1e-8)

        self.assertTrue(yolo_adapter_data_transform.frame_index, 3)

    def test_transform_3_frames_of_1920_1080_into_128_128_tensors(self):
        ndarray_to_tensor = NDArrayToTensor(size=(128, 128))

        decord_array = decord.ndarray.array(np.zeros((3, 1920, 1080, 3)))

        tensors = ndarray_to_tensor(decord_array)

        self.assertTrue(isinstance(tensors, torch.Tensor))
        self.assertTrue(tensors.shape == (3, 3, 128, 128))

    def test_transform_3_frames_of_1920_1080_into_128_128_list_of_numpy_arrays_uint8(self):
        ndarray_to_image = NDArrayToImage(size=(128, 128))

        decord_array = decord.ndarray.array(np.zeros((3, 1920, 1080, 3)))

        list_of_images = ndarray_to_image(decord_array)

        self.assertTrue(isinstance(list_of_images, list))
        self.assertTrue(len(list_of_images) == 3)
        self.assertTrue(list_of_images[0].shape == (128, 128, 3))
        self.assertTrue(list_of_images[1].shape == (128, 128, 3))
        self.assertTrue(list_of_images[2].shape == (128, 128, 3))
