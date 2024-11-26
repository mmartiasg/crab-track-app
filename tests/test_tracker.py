import unittest

import pandas as pd
import ultralytics.engine.results

from src.callbacks.post_processing import CallbackInterpolateCoordinates, CallbackSaveToDisk, \
    CallbackDenormalizeCoordinates
from src.tracking.yolo import TrackerByDetection, TrackerByByteTrack
import os
from src.utils.constants import Config
from ultralytics.engine.results import Results
import torch
import numpy as np
from src.transforms.Adapter import YoloAdapter, DefaultAdapter
from unittest.mock import patch, MagicMock


class TestTracker(unittest.TestCase):
    @patch("src.tracking.yolo.VideoFramesGenerator")
    def setUp(self, video_frames_generator_mock_class):
        super(TestTracker, self).setUp()

        self.config = Config(config_file_path=os.path.join(os.path.dirname(__file__),
                                                           'test_run_conf.yaml'))
        self.video_frame_size = (self.config.get_config["input"]["resolution"]["width"],
                                 self.config.get_config["input"]["resolution"]["height"])
        self.frame_original_size = (1920, 1080)
        self.coordinate_columns = ["pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"]
        self.test_videos_output_path = os.path.join(os.path.dirname(__file__),
                                                    self.config.get_config["output"]["path"],
                                                    "videos")
        self.test_stats_output_path = os.path.join(os.path.dirname(__file__),
                                                   self.config.get_config["output"]["path"],
                                                   "stats")
        self.test_logs_output_path = os.path.join(os.path.dirname(__file__), self.config.get_config["output"]["path"],
                                                  "logs")
        self.frame_original_size = (1920, 1080)
        self.coordinate_columns = ["x1", "y1", "x2", "y2"]

        os.makedirs(self.test_logs_output_path, exist_ok=True)

        self.test_video_1_path = os.path.join(os.path.dirname(__file__),
                                              self.config.get_config["input"]["path"],
                                              "test_sample_1_720p.mp4")

        # VideoFramesGenerator is a generator iter returns self but in this case I have to return the same as next
        video_frames_generator_mock_instance = video_frames_generator_mock_class.return_value
        video_frames_generator_mock_instance.__next__.return_value = np.zeros((3, 3, 256, 256))
        video_frames_generator_mock_instance.__len__.return_value = 3
        video_frames_generator_mock_instance.__iter__.return_value = np.zeros((3, 3, 256, 256))

        self.tracker = TrackerByDetection(input_video_path=self.test_video_1_path,
                                          batch_size=self.config.get_config["model"]["batch_size"],
                                          internal_resolution=(
                                              self.config.get_config["model"]["internal_resolution"]["height"],
                                              self.config.get_config["model"]["internal_resolution"]["width"]),
                                          confidence_threshold=self.config.get_config["model"]["conf_threshold"],
                                          nms_threshold=self.config.get_config["model"]["nms_threshold"],
                                          device=self.config.get_config["model"]["device"],
                                          model_weights=self.config.get_config["model"]["path"],
                                          response_transform=YoloAdapter(video_name="test", tracker_name="yolov8",
                                                                         dataset="test"),
                                          log_dir=os.path.join(os.path.dirname(__file__),
                                                               self.config.get_config["output"]["path"]))

        self.tracker_raw_response = TrackerByByteTrack(input_video_path=self.test_video_1_path,
                                                       batch_size=self.config.get_config["model"]["batch_size"],
                                                       internal_resolution=(
                                                           self.config.get_config["model"]["internal_resolution"][
                                                               "height"],
                                                           self.config.get_config["model"]["internal_resolution"][
                                                               "width"]),
                                                       confidence_threshold=self.config.get_config["model"][
                                                           "conf_threshold"],
                                                       nms_threshold=self.config.get_config["model"]["nms_threshold"],
                                                       device=self.config.get_config["model"]["device"],
                                                       model_weights=self.config.get_config["model"]["path"],
                                                       response_transform=DefaultAdapter(video_name="test",
                                                                                         tracker_name="test",
                                                                                         dataset="test"),
                                                       log_dir=os.path.join(os.path.dirname(__file__),
                                                                            self.config.get_config["output"]["path"]))

        results1 = Results(boxes=torch.tensor([[256, 256, 115, 250, 0, 0.9]]),
                           path=".", names={0: 'Crab', 1: 'Backgroud'},
                           orig_img=np.zeros((1920, 1080)))

        results1.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                          'preprocess': 0.00035390257835388184}

        self.tracker_raw_response.model.predict = MagicMock(return_value=[results1])
        self.tracker.model.predict = MagicMock(return_value=[results1])

    def test_one_step_prediction_returns_one_bbox_from_class_Results(self):
        batches = [torch.zeros(3, 256, 256)]

        for batch in batches:
            result = self.tracker_raw_response.predict_step(batch)

        self.assertTrue(isinstance(result[0][0], ultralytics.engine.results.Results))

    def test_one_step_prediction_returns_one_bbox_normalized_0237_0133_0165_01302(self):
        batches = [torch.zeros(3, 256, 256)]

        for batch in batches:
            result = self.tracker.predict_step(batch)

        self.assertAlmostEqual(round(result[0][0]["x1"], 4), round(256 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0]["y1"], 4), round(256 / 1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0]["x2"], 4), round(115 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0]["y2"], 4), round(250 / 1920, 4), delta=1e-7)

    def test_flatten_dictionary_array(self):
        from src.tracking.yolo import flatten_list

        # 3 frames 2 with 3 bboxes and last one with one
        nested_list = [[{"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9},
                        {"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9},
                        {"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9}],
                       [{"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9},
                        {"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9},
                        {"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9}],
                       [{"x1": 0.9, "y1": 0.9, "x2": 0.9, "y2": 0.9}]]

        flatten_list = flatten_list(nested_list)

        self.assertTrue(len(nested_list) == 3)
        self.assertTrue(len(flatten_list) == 7)

    @patch("src.tracking.yolo.VideoFramesGenerator")
    def test_create_pandas_dataframe_with_3_records_from_record1_bbox_normalized_0237_0133_0165_01302_floats(self,
                                                                                                             video_frames_generator_mock_class):

        # VideoFramesGenerator is a generator iter returns self but in this case I have to return the same as next
        video_frames_generator_mock_instance = video_frames_generator_mock_class.return_value
        video_frames_generator_mock_instance.__next__.return_value = np.zeros((3, 3, 256, 256))
        video_frames_generator_mock_instance.__len__.return_value = 3
        video_frames_generator_mock_instance.__iter__.return_value = np.zeros((3, 3, 256, 256))

        results1 = Results(boxes=torch.tensor([256, 256, 115, 250, 0, 0.9]),
                           path=".", names={0: 'Crab', 1: 'Backgroud'},
                           orig_img=np.zeros((1920, 1080)))

        results1.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                          'preprocess': 0.00035390257835388184}

        # 1 bbox per frame
        self.tracker.model.predict = MagicMock(return_value=[results1])

        preds_df = self.tracker.track_video(callbacks=None)

        self.assertTrue(isinstance(preds_df, pd.DataFrame))
        self.assertTrue("x1" in preds_df.columns)
        self.assertTrue("y1" in preds_df.columns)
        self.assertTrue("x2" in preds_df.columns)
        self.assertTrue("y2" in preds_df.columns)

        self.assertTrue(len(preds_df) == 3)

        self.assertTrue(preds_df["x1"].dtype == float)
        self.assertTrue(preds_df["y1"].dtype == float)
        self.assertTrue(preds_df["x2"].dtype == float)
        self.assertTrue(preds_df["y2"].dtype == float)

        self.assertAlmostEqual(round(preds_df.iloc[0]["x1"], 4), round(256 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0]["y1"], 4), round(256 / 1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0]["x2"], 4), round(115 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0]["y2"], 4), round(250 / 1920, 4), delta=1e-7)

    @patch("src.tracking.yolo.VideoFramesGenerator")
    def test_call_chain_callbacks_interpolation_and_denormalization_bbox_normalized_returns_256_256_115_249_file_saved_as_test_post_processed_csv(self,
                                                                                                 video_frames_generator_mock_class):

        coordinates_columns = ["x1", "y1", "x2", "y2"]

        callback_list = [
            CallbackInterpolateCoordinates(
                coordinates_columns=coordinates_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=coordinates_columns,
                image_size=(self.config.get_config["input"]["resolution"]["width"],
                            self.config.get_config["input"]["resolution"]["height"]),
                method="xyxy"),
            CallbackSaveToDisk(file_path=os.path.join(self.test_stats_output_path,
                                                      "test" + "_post_processed.csv"))
        ]

        # VideoFramesGenerator is a generator iter returns self but in this case I have to return the same as next
        video_frames_generator_mock_instance = video_frames_generator_mock_class.return_value
        video_frames_generator_mock_instance.__next__.return_value = np.zeros((3, 3, 256, 256))
        video_frames_generator_mock_instance.__len__.return_value = 3
        video_frames_generator_mock_instance.__iter__.return_value = np.zeros((3, 3, 256, 256))

        results1 = Results(boxes=torch.tensor([256, 256, 115, 250, 0, 0.9]),
                           path=".", names={0: 'Crab', 1: 'Backgroud'},
                           orig_img=np.zeros((720, 1280)))

        results1.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                          'preprocess': 0.00035390257835388184}

        # 1 bbox per frame
        self.tracker.model.predict = MagicMock(return_value=[results1])

        preds_df = self.tracker.track_video(callbacks=callback_list)

        self.assertTrue(isinstance(preds_df, pd.DataFrame))
        self.assertTrue("x1" in preds_df.columns)
        self.assertTrue("y1" in preds_df.columns)
        self.assertTrue("x2" in preds_df.columns)
        self.assertTrue("y2" in preds_df.columns)

        self.assertTrue(len(preds_df) == 3)

        self.assertTrue(preds_df["x1"].dtype == int)
        self.assertTrue(preds_df["y1"].dtype == int)
        self.assertTrue(preds_df["x2"].dtype == int)
        self.assertTrue(preds_df["y2"].dtype == int)

        self.assertEqual(preds_df.iloc[0]["x1"], 256)
        self.assertEqual(preds_df.iloc[0]["y1"], 256)
        self.assertEqual(preds_df.iloc[0]["x2"], 115)
        self.assertEqual(preds_df.iloc[0]["y2"], 249)

        self.assertTrue(os.path.exists(os.path.join(self.test_stats_output_path,
                                                      "test" + "_post_processed.csv")))
