from typing import List

from pandas.errors import IntCastingNaNError
from ultralytics import YOLO
from src.dataloaders.video_loader import VideoFramesGenerator
import logging
import os
import itertools
import pandas as pd
from src.transforms.Input import NDArrayToTensor, NDArrayToImage


def setup_logging(log_dir, video_name):
    tracker_logging = logging.getLogger(__name__)
    logger_file_handler = logging.FileHandler(
        os.path.join(log_dir, "logs", f"tracker_video_{video_name}.log"),
        mode="w",
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {filename} - {funcName} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_file_handler.setFormatter(formatter)
    tracker_logging.setLevel(logging.DEBUG)
    tracker_logging.addHandler(logger_file_handler)

    return tracker_logging


def flatten_list(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


class BaseTracker:
    def __init__(self,
                 input_video_path,
                 batch_size,
                 confidence_threshold,
                 nms_threshold,
                 device,
                 model_weights,
                 response_transform,
                 coordinates_columns,
                 internal_resolution,
                 log_dir,
                 threads_per_video=1):

        self.input_video_path = input_video_path
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.threads_per_video = threads_per_video
        self.model = YOLO(model_weights,
                          task="detect",
                          verbose=False)
        self.response_transform = response_transform
        self.video_name = input_video_path.split("/")[-1].split(".")[0]
        self.internal_resolution = internal_resolution
        self.logger = setup_logging(log_dir=log_dir, video_name=self.video_name)
        self.loader = self.set_up_video_loader()
        self.coordinates_columns = coordinates_columns

    def predict_step(self, batch):
        raise NotImplementedError("Implement in subclass")

    def set_up_video_loader(self):
        raise NotImplementedError("Implement in subclass")

    def track_video(self, callbacks=None):
        if callbacks is None:
            callbacks = []

        predictions = []
        for batch in self.loader:
            pred = self.predict_step(batch)
            predictions.extend(pred)
            self.logger.debug(f"""
                        Frames processed: {self.response_transform.frame_index} |
                        Frames with prediction over ({self.confidence_threshold}): {self.response_transform.frames_with_measurement} |
                        Frames without prediction ({self.confidence_threshold}): {self.response_transform.frames_without_measurement} |
                        Process time total: {round(self.response_transform.inference_time_batch, 4)} ms, 
                        per frame: {round(self.response_transform.inference_time_sample, 4)} ms
                        """)

        self.logger.info(f"Finished tracking {self.video_name}")

        predictions_df = self.create_coordinates_dataframe(flatten_list(predictions))

        self.post_process(predictions_df, callbacks)

        self.clean_up()

        return predictions_df

    def post_process(self, predictions, callbacks):
        for callback in callbacks:
            self.logger.debug(f"Start executing callback: {callback.__name__()}")
            try:
                callback(predictions)
            except IntCastingNaNError as e:
                self.logger.error(f"Fail executing callback: {callback.__name__()}: {e}", exc_info=True)
            except Exception as e:
                self.logger.error(f"Fail executing callback: {callback.__name__()}: {e}", exc_info=True)

            self.logger.debug(f"Finish executing callback: {callback.__name__()}")

    def transform_data_response(self, response):
        return self.response_transform(response)

    def create_coordinates_dataframe(self, tracker_stats_list: List) -> pd.DataFrame:
        stats_df = None

        try:
            stats_df = pd.DataFrame(tracker_stats_list)
            stats_df = stats_df.astype(dict(zip(self.coordinates_columns, [float, float, float, float])))
        except Exception as e:
            self.logger.critical("Building dataframe", exc_info=True)

        return stats_df

    def clean_up(self):
        self.logger.info("Free resources allocated")
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
            del handler


class TrackerByDetection(BaseTracker):
    def predict_step(self, batch):
        return self.transform_data_response(self.model.predict(batch,
                                                               stream=True,
                                                               conf=self.confidence_threshold,
                                                               iou=self.nms_threshold))

    def set_up_video_loader(self):
        video_frame_transform = NDArrayToTensor(self.internal_resolution)
        loader = VideoFramesGenerator(video_path=self.input_video_path,
                                      transform=video_frame_transform,
                                      batch_size=self.batch_size,
                                      num_threads=self.threads_per_video)
        self.logger.info(f"Video {self.video_name} loaded with frames: {loader.__len__()}")
        return loader


class TrackerByByteTrack(BaseTracker):
    def predict_step(self, batch):
        return self.transform_data_response(self.model.track(batch,
                                                             stream=True,
                                                             persist=True,
                                                             conf=self.confidence_threshold,
                                                             iou=self.nms_threshold))

    def set_up_video_loader(self):
        video_frame_transform = NDArrayToImage(self.internal_resolution)
        loader = VideoFramesGenerator(video_path=self.input_video_path,
                                      transform=video_frame_transform,
                                      batch_size=self.batch_size,
                                      num_threads=self.threads_per_video)
        self.logger.info(f"Video {self.video_name} loaded with frames: {loader.__len__()}")
        return loader
