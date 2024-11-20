import logging
import pandas as pd
from ultralytics import YOLO
import os
from src.dataloaders.video_loader import VideoFramesGenerator
import torchvision
from src.transforms.Adapter import YoloAdapter
import multiprocessing as mpt
import itertools


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


def create_coordinates_dataframe(tracker_stats_list, logger):
    stats_df = None
    try:
        stats_df = pd.DataFrame(flatten_list(tracker_stats_list))
        stats_df = stats_df.astype({"x1": float, "y1": float, "x2": float, "y2": float})
    except Exception as e:
        logger.critical("Building dataframe", exc_info=True)

    return stats_df


# TODO: Refactor this into a class tracker with methods\
#  set-up, track_step, track_videp and several hooks to call callbacks
def track_object_v2(input_video_path,
                    out_path,
                    video_name,
                    tracker_name,
                    model_weights,
                    device="cpu",
                    batch_size=32,
                    confidence_threshold=0.8,
                    nms_threshold=0.5):

    # Set up logger
    tracker_logging = setup_logging(out_path, video_name)
    tracker_logging.info(f"Start tracking {video_name}")

    video_frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # TODO: Move to config?
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    # TODO: move to config
    loader = VideoFramesGenerator(video_path=input_video_path,
                                  transform=video_frame_transform,
                                  batch_size=batch_size,
                                  num_threads=mpt.cpu_count()//8)

    tracker_logging.debug(f"Video {video_name} loaded with frames: {loader.__len__()}")

    # Instance model
    model = YOLO(model_weights, task="detect", verbose=False)

    coordinates_bbox_list = []

    # TODO: move dataset name to config
    data_adapter = YoloAdapter(video_name=video_name, tracker_name=tracker_name, dataset="ICMAN3OCT2022")

    for batch in loader:
        # TODO: Refactor this in 2 tracker classes!
        batch_results = model.predict(batch,
                                      conf=confidence_threshold,
                                      iou=nms_threshold,
                                      verbose=False,
                                      device=device,
                                      stream=True
                                      )

        records = data_adapter(batch_results)
        coordinates_bbox_list.extend(records)

        tracker_logging.debug(f"""
                    Frames processed: {data_adapter.frame_index} |
                    Frames with prediction over ({confidence_threshold}): {data_adapter.frames_with_measurement} |
                    Frames without prediction ({confidence_threshold}): {data_adapter.frames_without_measurement} |
                    Process time total: {round(data_adapter.inference_time_batch, 4)} ms, 
                    per frame: {round(data_adapter.inference_time_sample, 4)} ms
                    """)

    tracker_logging.info(f"Finished tracking {video_name}")
    coordinates_df = create_coordinates_dataframe(coordinates_bbox_list, tracker_logging)

    tracker_logging.info("Free resources allocated")
    for handler in tracker_logging.handlers:
        handler.close()
        tracker_logging.removeHandler(handler)
        del handler

    del tracker_logging
    del loader
    del model
    del data_adapter

    return {
        "video_name": video_name,
        "coordinates": coordinates_df
    }


def track_object_v3(input_video_path,
                    out_path,
                    video_name,
                    tracker_name,
                    model_weights,
                    device="cpu",
                    batch_size=32,
                    confidence_threshold=0.8,
                    nms_threshold=0.5):

    # Set up logger
    tracker_logging = setup_logging(out_path, video_name)
    tracker_logging.info(f"Start tracking {video_name}")

    video_frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    # None is to be used with the yolo tracker
    loader = VideoFramesGenerator(video_path=input_video_path,
                                  transform=None,
                                  batch_size=batch_size,
                                  num_threads=mpt.cpu_count()//8)

    tracker_logging.debug(f"Video {video_name} loaded with frames: {loader.__len__()}")

    # Instance model
    model = YOLO(model_weights, task="detect", verbose=False)

    coordinates_bbox_list = []

    # TODO: move dataset name to config
    data_adapter = YoloAdapter(video_name=video_name, tracker_name=tracker_name, dataset="ICMAN3OCT2022")

    for batch in loader:
        # TODO: Refactor this in 2 tracker classes!
        # TODO: add option in confing to select tracker.
        # Track works with raw images
        # not sure how to feed it using a dataloader.
        batch_results = model.track(batch,
                                    persist=True,
                                    conf=confidence_threshold,
                                    iou=nms_threshold,
                                    verbose=False,
                                    device=device,
                                    stream=True)

        records = data_adapter(batch_results)
        coordinates_bbox_list.extend(records)

        tracker_logging.debug(f"""
                    Frames processed: {data_adapter.frame_index} |
                    Frames with prediction over ({confidence_threshold}): {data_adapter.frames_with_measurement} |
                    Frames without prediction ({confidence_threshold}): {data_adapter.frames_without_measurement} |
                    Process time total: {round(data_adapter.inference_time_batch, 4)} ms, 
                    per frame: {round(data_adapter.inference_time_sample, 4)} ms
                    """)

    tracker_logging.info(f"Finished tracking {video_name}")

    tracker_logging.info("Free resources allocated")
    for handler in tracker_logging.handlers:
        handler.close()
        tracker_logging.removeHandler(handler)
        del handler

    coordinates_df = create_coordinates_dataframe(coordinates_bbox_list, tracker_logging)

    del tracker_logging
    del loader
    del model

    return {
        "video_name": video_name,
        "coordinates": coordinates_df
    }
