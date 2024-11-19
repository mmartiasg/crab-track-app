import unittest
from src.transforms.Adapter import YoloAdapter
from ultralytics.engine.results import Results
import numpy as np
import torch


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
        self.assertAlmostEqual(round(json_response_batch[0][0]["pred_bbox_x1"], 4), round(256/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["pred_bbox_y1"], 4), round(256/1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["pred_bbox_x2"], 4), round(115/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[0][0]["pred_bbox_y2"], 4), round(250/1920, 4), delta=1e-7)

        self.assertAlmostEqual(round(json_response_batch[1][0]["pred_bbox_x1"], 4), round(256/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["pred_bbox_y1"], 4), round(256/1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["pred_bbox_x2"], 4), round(115/1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(json_response_batch[1][0]["pred_bbox_y2"], 4), round(250/1920, 4), delta=1e-7)

        self.assertAlmostEqual(json_response_batch[2][0]["pred_bbox_x1"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["pred_bbox_y1"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["pred_bbox_x2"], None)
        self.assertAlmostEqual(json_response_batch[2][0]["pred_bbox_y2"], None)

        self.assertAlmostEqual(yolo_adapter_data_transform.inference_time_batch, 5.405731499195099 * 3, delta=1e-8)
        self.assertAlmostEqual(yolo_adapter_data_transform.inference_time_batch / 3,
                               yolo_adapter_data_transform.inference_time_sample, delta=1e-8)

        self.assertTrue(yolo_adapter_data_transform.frame_index, 3)

