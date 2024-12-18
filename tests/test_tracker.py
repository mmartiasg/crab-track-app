import unittest

import pandas as pd
import ultralytics.engine.results

from src.callbacks.post_processing import CallbackInterpolateCoordinatesSingleObjectTracking, CallbackSaveToDisk, \
    CallbackDenormalizeCoordinates
from src.tracking.yolo import TrackerByDetection, TrackerByByteTrack, BaseTracker
import os
from src.utils.constants import Config
from ultralytics.engine.results import Results
import torch
import numpy as np
from src.transforms.Adapter import YoloAdapter, DefaultAdapter
from unittest.mock import patch, MagicMock, call
from src.tracking.yolo import flatten_list


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
        self.coordinate_columns = self.config.get_config["output"]["coordinates_columns"]

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
                                          coordinates_columns=self.coordinate_columns,
                                          response_transform=YoloAdapter(video_name="test",
                                                                         tracker_name="yolov8",
                                                                         dataset="test",
                                                                         column_names=self.config.get_config["output"]["coordinates_columns"]),
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
                                                       coordinates_columns=self.coordinate_columns,
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

        self.assertAlmostEqual(round(result[0][0][self.coordinate_columns[0]], 4), round(256 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0][self.coordinate_columns[1]], 4), round(256 / 1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0][self.coordinate_columns[2]], 4), round(115 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(result[0][0][self.coordinate_columns[3]], 4), round(250 / 1920, 4), delta=1e-7)

    def test_flatten_dictionary_array(self):
        from src.tracking.yolo import flatten_list

        # 3 frames 2 with 3 bboxes and last one with one
        nested_list = [[{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}],
                       [{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}],
                       [{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}]]

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
        self.assertTrue(self.coordinate_columns[0] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[1] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[2] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[3] in preds_df.columns)

        self.assertTrue(len(preds_df) == 3)

        self.assertTrue(preds_df[self.coordinate_columns[0]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[1]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[2]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[3]].dtype == float)

        self.assertAlmostEqual(round(preds_df.iloc[0][self.coordinate_columns[0]], 4), round(256 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0][self.coordinate_columns[1]], 4), round(256 / 1920, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0][self.coordinate_columns[2]], 4), round(115 / 1080, 4), delta=1e-7)
        self.assertAlmostEqual(round(preds_df.iloc[0][self.coordinate_columns[3]], 4), round(250 / 1920, 4), delta=1e-7)

    @patch("src.tracking.yolo.VideoFramesGenerator")
    def test_call_chain_callbacks_interpolation_and_denormalization_bbox_normalized_returns_256_256_115_249_file_saved_as_test_post_processed_csv(
            self,
            video_frames_generator_mock_class):

        callback_list = [
            CallbackInterpolateCoordinatesSingleObjectTracking(
                coordinates_columns=self.coordinate_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=self.coordinate_columns,
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
        self.assertTrue(self.coordinate_columns[0] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[1] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[2] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[3] in preds_df.columns)

        self.assertTrue(len(preds_df) == 3)

        self.assertTrue(preds_df[self.coordinate_columns[0]].dtype == int)
        self.assertTrue(preds_df[self.coordinate_columns[1]].dtype == int)
        self.assertTrue(preds_df[self.coordinate_columns[2]].dtype == int)
        self.assertTrue(preds_df[self.coordinate_columns[3]].dtype == int)

        self.assertEqual(preds_df.iloc[0][self.coordinate_columns[0]], 256)
        self.assertEqual(preds_df.iloc[0][self.coordinate_columns[1]], 256)
        self.assertEqual(preds_df.iloc[0][self.coordinate_columns[2]], 115)
        self.assertEqual(preds_df.iloc[0][self.coordinate_columns[3]], 249)

        self.assertTrue(os.path.exists(os.path.join(self.test_stats_output_path,
                                                    "test" + "_post_processed.csv")))

    @patch("src.tracking.yolo.YOLO")
    def test_instantiate_base_tracker_predict_step_and_set_up_video_loader_returns_NotImplementedError_exception(self,
                                                                                                                 yolo_class_mock):
        instance_yolo = yolo_class_mock.return_value

        self.assertRaises(NotImplementedError, BaseTracker, input_video_path=self.test_video_1_path,
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
                          model_weights="",
                          coordinates_columns=self.coordinate_columns,
                          response_transform=DefaultAdapter(video_name="test",
                                                            tracker_name="test",
                                                            dataset="test"),
                          log_dir=os.path.join(os.path.dirname(__file__),
                                               self.config.get_config["output"]["path"]))

        # This try is needed due to the setup_video method blows up during creation.
        # Thus, I need to capture that ignore it and use the other method that raise an exception predict_step
        try:
            base_tracker = BaseTracker(input_video_path = self.test_video_1_path,
            batch_size = self.config.get_config["model"]["batch_size"],
            internal_resolution = (
                self.config.get_config["model"]["internal_resolution"][
                    "height"],
                self.config.get_config["model"]["internal_resolution"][
                    "width"]),
            confidence_threshold = self.config.get_config["model"][
                "conf_threshold"],
            nms_threshold = self.config.get_config["model"]["nms_threshold"],
            device = self.config.get_config["model"]["device"],
            model_weights = "",
            coordinates_columns=self.coordinate_columns,
            response_transform = DefaultAdapter(video_name="test",
                                                tracker_name="test",
                                                dataset="test"),
            log_dir = os.path.join(os.path.dirname(__file__),
                                   self.config.get_config["output"]["path"]))

            self.assertRaises(NotImplementedError, base_tracker.predict_step, [])
        except NotImplementedError:
            pass

    def test_post_process_yolo_tracker_fails_with_log_error_entries_for_each_callback(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=["x3", "x4", "x5"],
            method="linear",
            max_distance=25)

        denormalize_callback = CallbackDenormalizeCoordinates(
            coordinates_columns=["x3", "x4", "x5"],
            image_size=(1920, 1080),
            method="xyxy")

        results1 = Results(boxes=torch.tensor([np.inf, np.nan, np.inf, np.inf, 0, 0.9]),
                           path=".", names={0: 'Crab', 1: 'Backgroud'},
                           orig_img=np.zeros((720, 1280)))

        results1.speed = {'inference': 5.405731499195099, 'postprocess': 0.4694610834121704,
                          'preprocess': 0.00035390257835388184}

        self.tracker.logger = MagicMock()

        # 1 bbox per frame
        self.tracker.model.predict = MagicMock(return_value=[results1])

        self.tracker.track_video([interpolate_callback, denormalize_callback])

        self.tracker.logger.error.assert_has_calls([
            call('Fail executing callback: CallbackInterpolateCoordinatesSingleObjectTracking: "None of [Index([\'x3\', \'x4\', \'x5\'], dtype=\'object\')] are in the [columns]"', exc_info=True),
            call('Fail executing callback: CallbackDenormalizeCoordinates: "None of [Index([\'x3\', \'x4\', \'x5\'], dtype=\'object\')] are in the [columns]"', exc_info=True)
        ])

    def test_create_coordinates_fails_with_critical_log(self):
        self.tracker.logger = MagicMock()

        dummy_data = {
            self.coordinate_columns[0]: [1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan],
            self.coordinate_columns[1]: [1, 1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],
            self.coordinate_columns[2]: [1, 1, 3, np.nan, np.nan, np.nan, np.nan, np.nan],
            self.coordinate_columns[3]: [1, 1, 4, np.nan, np.nan, np.nan, np.nan, np.nan]
        }

        self.tracker.create_coordinates_dataframe([dummy_data])

        self.tracker.logger.critical.assert_has_calls([
            call("Building dataframe", exc_info=True),
        ])

    def test_create_coordinates_returns_dataframe_with_4_columns_and_4_rows_from_coordinates_columns_in_conf_test(self):
        self.tracker.logger = MagicMock()

        # 3 frames 2 with 3 bboxes and last one with one
        dummy_data = [[{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}],
                       [{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9},
                        {self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}],
                       [{self.coordinate_columns[0]: 0.9, self.coordinate_columns[1]: 0.9, self.coordinate_columns[2]: 0.9, self.coordinate_columns[3]: 0.9}]]

        preds_df = self.tracker.create_coordinates_dataframe(flatten_list(dummy_data))

        self.assertTrue(isinstance(preds_df, pd.DataFrame))
        self.assertTrue(self.coordinate_columns[0] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[1] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[2] in preds_df.columns)
        self.assertTrue(self.coordinate_columns[3] in preds_df.columns)

        self.assertTrue(len(preds_df) == 7)

        self.assertTrue(preds_df[self.coordinate_columns[0]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[1]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[2]].dtype == float)
        self.assertTrue(preds_df[self.coordinate_columns[3]].dtype == float)

        self.assertEqual(0.9, preds_df.iloc[0][self.coordinate_columns[0]])
        self.assertEqual(0.9, preds_df.iloc[0][self.coordinate_columns[1]])
        self.assertEqual(0.9, preds_df.iloc[0][self.coordinate_columns[2]])
        self.assertEqual(0.9, preds_df.iloc[0][self.coordinate_columns[3]])
