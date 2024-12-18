import unittest

import numpy as np

from src.utils.constants import Config
import pandas as pd
from src.callbacks.compose import ComposeCallback
from src.callbacks.post_processing import (CallbackInterpolateCoordinatesSingleObjectTracking,
                                           CallbackDenormalizeCoordinates,
                                           CallbackSaveToDisk)
from src.callbacks.video_render import CallbackRenderVideoTracking
import os
import shutil


class DataloaderSuitCase(unittest.TestCase):
    def setUp(self):
        super(DataloaderSuitCase, self).setUp()
        self.config = Config(config_file_path=os.path.join(os.path.dirname(__file__),
                                                           'test_run_conf.yaml'))
        self.video_frame_size = (self.config.get_config["input"]["resolution"]["width"],
                                 self.config.get_config["input"]["resolution"]["height"])
        self.frame_original_size = (1920, 1080)
        self.coordinate_columns = ["x1", "y1", "x2", "y2"]
        self.test_videos_output_path = os.path.join(os.path.dirname(__file__),
                                                    self.config.get_config["output"]["path"],
                                                    "videos")
        self.test_stats_output_path = os.path.join(os.path.dirname(__file__),
                                                    self.config.get_config["output"]["path"],
                                                    "stats")

        shutil.rmtree(self.test_videos_output_path, ignore_errors=True)
        shutil.rmtree(self.test_stats_output_path, ignore_errors=True)

        os.makedirs(self.test_videos_output_path, exist_ok=True)
        os.makedirs(self.test_stats_output_path, exist_ok=True)

    def test_interpolate_coordinates_from_sample_2_coordinates_to_frame_0_from_frame_4(self):
        sample_2_coordinates_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                           "test_data/sample_2_coordinates.csv"))

        interpolated_callback = CallbackInterpolateCoordinatesSingleObjectTracking(coordinates_columns=self.coordinate_columns,
                                                                                   method="linear",
                                                                                   max_distance=25)

        interpolated_coordinates_df = interpolated_callback(sample_2_coordinates_df)

        self.assertTrue(interpolated_coordinates_df.iloc[0][self.coordinate_columns]["x1"],
                        interpolated_coordinates_df.iloc[3][self.coordinate_columns]["x1"])

    def test_denormalized_coordinates_from_sample_1_first_frame_is_1064_85_1343_232(self):
        sample_1_coordinates = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                        "test_data/sample_1_coordinates.csv"))

        denormalized_callback = CallbackDenormalizeCoordinates(coordinates_columns=self.coordinate_columns,
                                                               method="xyxy",
                                                               image_size=(1920, 1080))

        denormalized_coordinates = denormalized_callback(sample_1_coordinates).iloc[0]

        self.assertEqual(denormalized_coordinates[self.coordinate_columns[0]], 1064)
        self.assertEqual(denormalized_coordinates[self.coordinate_columns[1]], 85)
        self.assertEqual(denormalized_coordinates[self.coordinate_columns[2]], 1343)
        self.assertEqual(denormalized_coordinates[self.coordinate_columns[3]], 232)

    def test_compose_callbacks_using_sample_2_returns_denormalized_coordinates_1289_985_1560_1062_saves_csv_file(
            self):
        sample_2_coordinates_df = pd.read_csv(
                                    os.path.join(
                                        os.path.dirname(__file__),
                                        "test_data/sample_2_coordinates.csv")
                                )

        r = {"video_name": "sample_2", "coordinates": sample_2_coordinates_df}

        compose_callback = ComposeCallback([
            CallbackInterpolateCoordinatesSingleObjectTracking(
                coordinates_columns=self.coordinate_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=self.coordinate_columns,
                image_size=(1920, 1080),
                method="xyxy"),
            CallbackSaveToDisk(file_path=os.path.join(
                                    self.test_stats_output_path,
                                    r["video_name"] + "_post_processed.csv"
                                )
            )
        ])

        sample_2_post_processed = compose_callback(sample_2_coordinates_df).iloc[0]

        self.assertEqual(sample_2_post_processed[self.coordinate_columns[0]], 1289)
        self.assertEqual(sample_2_post_processed[self.coordinate_columns[1]], 985)
        self.assertEqual(sample_2_post_processed[self.coordinate_columns[2]], 1560)
        self.assertEqual(sample_2_post_processed[self.coordinate_columns[3]], 1062)
        self.assertTrue(os.path.exists(
                                    os.path.join(
                                        self.test_stats_output_path,
                                        r["video_name"] + "_post_processed.csv"
                                    )
                        )
        )

    def test_export_video_with_boundary_and_path_for_sample_video_2_validate_if_file_exists_true(self):
        video_name = "sample_2_test_video"
        sample_2_coordinates_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                           "test_data/sample_2_coordinates.csv"))

        render_callback = CallbackRenderVideoTracking(output_video_path=os.path.join(self.test_videos_output_path,
                                                                                     video_name + ".mp4"),
                                                      input_video_path=os.path.join(os.path.dirname(__file__),
                                                                            self.config.get_config["input"]["path"],
                                                                            "test_sample_2_720p.mp4"),
                                                      coordinate_columns=self.coordinate_columns,
                                                      bbox_color=(0, 0, 255))
        compose_callback = ComposeCallback([
            CallbackInterpolateCoordinatesSingleObjectTracking(
                coordinates_columns=self.coordinate_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=self.coordinate_columns,
                image_size=self.video_frame_size,
                method="xyxy"),
        ])
        post_processed_coordinates_df = compose_callback(sample_2_coordinates_df)

        render_callback(post_processed_coordinates_df)

        self.assertTrue(os.path.exists(os.path.join(self.test_videos_output_path,
                                                    video_name + ".mp4"))
                        )

    def test_export_video_with_boundary_and_path_for_sample_video_1_validate_if_file_exists_true(self):
        video_name = "sample_1_test_video"
        sample_2_coordinates_df = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                           "test_data/sample_1_coordinates.csv"))

        render_callback = CallbackRenderVideoTracking(output_video_path=os.path.join(self.test_videos_output_path,
                                                                                     video_name + ".mp4"),
                                                      input_video_path=os.path.join(os.path.dirname(__file__),
                                                                            self.config.get_config["input"]["path"],
                                                                            "test_sample_1_720p.mp4"),
                                                      coordinate_columns=self.coordinate_columns,
                                                      bbox_color=(0, 0, 255))

        compose_callback = ComposeCallback([
            CallbackInterpolateCoordinatesSingleObjectTracking(
                coordinates_columns=self.coordinate_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=self.coordinate_columns,
                image_size=self.video_frame_size,
                method="xyxy"),
        ])
        post_processed_coordinates_df = compose_callback(sample_2_coordinates_df)

        render_callback(post_processed_coordinates_df)

        self.assertTrue(os.path.exists(os.path.join(self.test_videos_output_path,
                                                    video_name + ".mp4"))
        )

    def test_interpolate_with_max_distance_2_more_a_dataframe_with_4_nan_rows_returns_3_4_5_6_rows_nan(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=2)

        dummy_data = {
            'x1': [1, 1, 1, np.nan, np.nan, np.nan, np.nan, 1],
            'y1': [1, 1, 2, np.nan, np.nan, np.nan, np.nan, 1],
            'x2': [1, 1, 3, np.nan, np.nan, np.nan, np.nan, 1],
            'y2': [1, 1, 4, np.nan, np.nan, np.nan, np.nan, 1]
        }
        dummy_coordinates_df = pd.DataFrame(dummy_data)

        dummy_coordinates_interpolated_df = interpolate_callback(dummy_coordinates_df)

        self.assertTrue((dummy_coordinates_interpolated_df.index[dummy_coordinates_interpolated_df.isna().all(axis=1)]
                         == [3, 4, 5, 6]).all())

    def test_interpolate_with_max_distance_4_more_a_dataframe_with_4_nan_rows_returns_no_rows_nan(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=5)

        dummy_data = {
            'x1': [1, 1, 1, np.nan, np.nan, np.nan, np.nan, 1],
            'y1': [1, 1, 2, np.nan, np.nan, np.nan, np.nan, 1],
            'x2': [1, 1, 3, np.nan, np.nan, np.nan, np.nan, 1],
            'y2': [1, 1, 4, np.nan, np.nan, np.nan, np.nan, 1]
        }
        dummy_coordinates_df = pd.DataFrame(dummy_data)

        dummy_coordinates_interpolated_df = interpolate_callback(dummy_coordinates_df)

        self.assertTrue((dummy_coordinates_interpolated_df.index[dummy_coordinates_interpolated_df.isna().all(axis=1)]
                         == []).all())

    def test_interpolate_with_max_distance_4_more_a_dataframe_with_4_nan_rows_in_blocks_and_first_row_nan_returns_no_rows_nan_first_row_equals_1_1_1_1(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=5)

        dummy_data = {
            'x1': [np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, 1],
            'y1': [np.nan, 1, 2, np.nan, np.nan, np.nan, np.nan, 1],
            'x2': [np.nan, 1, 3, np.nan, np.nan, np.nan, np.nan, 1],
            'y2': [np.nan, 1, 4, np.nan, np.nan, np.nan, np.nan, 1]
        }
        dummy_coordinates_df = pd.DataFrame(dummy_data)

        dummy_coordinates_interpolated_df = interpolate_callback(dummy_coordinates_df)

        self.assertTrue((dummy_coordinates_interpolated_df.index[dummy_coordinates_interpolated_df.isna().all(axis=1)]
                         == []).all())

        self.assertTrue((dummy_coordinates_interpolated_df.iloc[0] == [1, 1, 1, 1]).all())

    def test_interpolate_with_max_distance_4_more_a_dataframe_with_4_nan_rows_in_blocks_and_first_row_nan_returns_no_rows_nan_last_row_equals_1_1_1_1(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=5)

        dummy_data = {
            'x1': [1, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
            'y1': [1, 1, 2, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
            'x2': [1, 1, 3, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
            'y2': [1, 1, 4, np.nan, np.nan, np.nan, np.nan, 1, np.nan]
        }
        dummy_coordinates_df = pd.DataFrame(dummy_data)

        dummy_coordinates_interpolated_df = interpolate_callback(dummy_coordinates_df)

        self.assertTrue((dummy_coordinates_interpolated_df.index[dummy_coordinates_interpolated_df.isna().all(axis=1)]
                         == []).all())

        self.assertTrue((dummy_coordinates_interpolated_df.iloc[-1] == [1, 1, 1, 1]).all())

    def test_interpolate_with_max_distance_4_more_a_dataframe_with_5_nan_rows_in_blocks_last_row_equals_1_2_3_4_last_non_nan_is_propagated_bfill(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=5)

        dummy_data = {
            'x1': [1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan],
            'y1': [1, 1, 2, np.nan, np.nan, np.nan, np.nan, np.nan],
            'x2': [1, 1, 3, np.nan, np.nan, np.nan, np.nan, np.nan],
            'y2': [1, 1, 4, np.nan, np.nan, np.nan, np.nan, np.nan]
        }
        dummy_coordinates_df = pd.DataFrame(dummy_data)

        dummy_coordinates_interpolated_df = interpolate_callback(dummy_coordinates_df)

        self.assertTrue((dummy_coordinates_interpolated_df.index[dummy_coordinates_interpolated_df.isna().all(axis=1)]
                         == []).all())

        self.assertTrue((dummy_coordinates_interpolated_df.iloc[-1] == [1, 2, 3, 4]).all())

    def test_callbacknaming_returns_CallbackInterpolateCoordinatesSingleObjectTracking(self):
        interpolate_callback = CallbackInterpolateCoordinatesSingleObjectTracking(
            coordinates_columns=self.coordinate_columns,
            method="linear",
            max_distance=5)
        self.assertEqual("CallbackInterpolateCoordinatesSingleObjectTracking", interpolate_callback.__name__())

    def test_callbacknaming_callback_compose_returns_CallbackInterpolateCoordinatesSingleObjectTracking_CallbackDenormalizeCoordinates(self):
        video_name = ""
        r = {"video_name": video_name}
        compose_callback = ComposeCallback([
            CallbackInterpolateCoordinatesSingleObjectTracking(
                coordinates_columns=self.coordinate_columns,
                method="linear",
                max_distance=25),
            CallbackDenormalizeCoordinates(
                coordinates_columns=self.coordinate_columns,
                image_size=(1920, 1080),
                method="xyxy"),
            CallbackRenderVideoTracking(output_video_path=os.path.join(self.test_videos_output_path,
                                                                       video_name + ".mp4"),
                                        input_video_path=os.path.join(os.path.dirname(__file__),
                                                                      self.config.get_config["input"]["path"],
                                                                      "test_sample_1_720p.mp4"),
                                        coordinate_columns=self.coordinate_columns,
                                        bbox_color=(0, 0, 255)),
            CallbackSaveToDisk(file_path=os.path.join(
                                    self.test_stats_output_path,
                                    r["video_name"] + "_post_processed.csv"
                                )
            )
        ])

        self.assertEqual("Compose callback: CallbackInterpolateCoordinatesSingleObjectTracking_CallbackDenormalizeCoordinates_CallbackRenderVideoTracking_CallbackSaveToDisk", compose_callback.__name__())
