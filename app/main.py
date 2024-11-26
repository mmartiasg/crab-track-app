#!/usr/bin/env python3
import argparse
import os
from src.transforms.Adapter import YoloAdapter
from src.utils.constants import Config
from joblib import Parallel, delayed, parallel_backend
import glob
import logging
from src.callbacks.compose import ComposeCallback
from src.callbacks.post_processing import CallbackDenormalizeCoordinates, CallbackInterpolateCoordinates, \
    CallbackSaveToDisk
from src.callbacks.video_render import CallbackRenderVideoSingleObjectTracking
from src.tracking import TRACKER_CLASSES
import pandas as pd


def create_job(video_path, config, logging):
    video_name = video_path.split("/")[-1].split(".")[0]

    # Get the tracker algorithm from the configuration
    algorithm_name = config.get_config["model"]["algorithm"]
    tracker_class = TRACKER_CLASSES.get(algorithm_name)
    if tracker_class is None:
        logging.critical(
            f"No tracker found for algorithm '{algorithm_name}'. Available options: {list(TRACKER_CLASSES.keys())}")
        raise ModuleNotFoundError(
            f"No tracker found for algorithm '{algorithm_name}'. Available options: {list(TRACKER_CLASSES.keys())}")

    render_videos = config.get_config["output"]["render_videos"]

    coordinates_columns = config.get_config["output"]["coordinates_columns"]

    callback_list = []
    postfix = ""

    if config.get_config["output"]["interpolate"]["enabled"]:
        callback_list.append(CallbackInterpolateCoordinates(
            coordinates_columns=coordinates_columns,
            method="linear",
            # 5 frames interpolation limit
            # 1 video has 25 frames per second thus 25 * 5
            max_distance=config.get_config["output"]["interpolate"]["max_distance"]))
        postfix += "_interpolated"

    if config.get_config["output"]["denormalize"]["enabled"]:
        callback_list.append(CallbackDenormalizeCoordinates(
            coordinates_columns=coordinates_columns,
            image_size=(config.get_config["input"]["resolution"]["width"],
                        config.get_config["input"]["resolution"]["height"]),
            method="xyxy"))
        postfix += "_denormalized"

    callback_list.append(CallbackSaveToDisk(file_path=os.path.join(config.get_config["output"]["path"],
                                                                   "stats",
                                                                   video_name + f"{postfix}.csv")))

    if video_name in render_videos:
        render_callback = CallbackRenderVideoSingleObjectTracking(
            output_video_path=os.path.join(config.get_config["output"]["path"],
                                           "videos",
                                           video_name + ".mp4"),
            input_video_path=os.path.join(config.get_config["input"]["path"],
                                          video_name + ".mp4"),
            coordinate_columns=coordinates_columns,
            bbox_color=(0, 0, 255))
        callback_list.append(render_callback)

    compose_callback = ComposeCallback(callback_list)
    save_raw_data_callback = CallbackSaveToDisk(file_path=os.path.join(config.get_config["output"]["path"],
                                                                       "stats",
                                                                       video_name + ".csv"))
    callbacks = [save_raw_data_callback, compose_callback]

    tracker = tracker_class(input_video_path=video_path,
                            log_dir=config.get_config["output"]["path"],
                            # keep the name of the video
                            batch_size=config.get_config["model"]["batch_size"],
                            confidence_threshold=config.get_config["model"]["conf_threshold"],
                            nms_threshold=config.get_config["model"]["nms_threshold"],
                            device=config.get_config["model"]["device"],
                            threads_per_video=config.get_config["multiprocess"]["threads_per_video"],
                            internal_resolution=(config.get_config["model"]["internal_resolution"]["height"],
                                                 config.get_config["model"]["internal_resolution"]["width"]),
                            model_weights=config.get_config["model"]["path"],
                            response_transform=YoloAdapter(video_name=video_path.split("/")[-1].split(".")[0],
                                                           tracker_name="yolov8",
                                                           dataset="test"))
    logging.info(f"Start tracking {video_name}")
    return tracker.track_video(callbacks=callbacks)


def render_video(video, stats_path, input_video_path, output_video_path, config):
    postfix = ""

    if config.get_config["output"]["interpolate"]["enabled"]:
        postfix += "_interpolated"

    if config.get_config["output"]["denormalize"]["enabled"]:
        postfix += "_denormalized"

    if video + f"{postfix}.csv" in os.listdir(stats_path):
        render_callback = CallbackRenderVideoSingleObjectTracking(
            output_video_path=os.path.join(output_video_path,
                                           video + ".mp4"),
            input_video_path=os.path.join(input_video_path,
                                          video + ".mp4"),
            coordinate_columns=["x1", "y1", "x2", "y2"],
            bbox_color=(0, 0, 255))

        render_callback(pd.read_csv(os.path.join(stats_path, video + "_post_processed.csv")))


def interpolate_coordinates(video, stats_path, coordinates_columns, method="linear", max_distance=25):
    if video + ".csv" in os.listdir(stats_path):
        interpolate_callback = CallbackInterpolateCoordinates(
            coordinates_columns=coordinates_columns,
            method=method,
            # 5 frames interpolation limit
            # 1 video has 25 frames per second thus 25 * 5
            max_distance=max_distance)

        interpolated_df = interpolate_callback(pd.read_csv(os.path.join(stats_path, video + ".csv")))
        interpolated_df.to_csv(f"{video}_interpolated.csv", index=False)


def denormalize_coordinates(video, stats_path, coordinates_columns, width, height):
    if video + ".csv" in os.listdir(stats_path):
        denormalize_callback = CallbackDenormalizeCoordinates(
            coordinates_columns=coordinates_columns,
            image_size=(width,
                        height),
            method="xyxy")

        denormalize_df = denormalize_callback(pd.read_csv(os.path.join(stats_path, video + ".csv")))
        denormalize_df.to_csv(f"{video}_denormalized.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Configurations for the script")
    parser.add_argument("--config_path",
                        help="Specify the config path",
                        default=None)
    parser.add_argument("--render_video_only",
                        help="Option to just render a video with an existing tracking file",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--interpolate_existing_tracks",
                        help="Interpolate existing tracks only",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--denormalized_existing_tracks",
                        help="Denormalized existing [interpolated] tracks only to output resolution in config",
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("--track",
                        help="Option track on the videos in the provided path",
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if args.config_path is not None:
        config = Config(args.config_path)
    else:
        raise (Exception("No config file specified"))

    if config.get_config["input"]["path"] is None:
        raise Exception("No input path specified")

    os.makedirs(os.path.join(config.get_config["output"]["path"], "logs"), exist_ok=True)

    # Set up logger
    main_logging = logging.getLogger(__name__)
    logger_file_handler = logging.FileHandler(
        os.path.join(config.get_config["output"]["path"], "logs", "main.log"),
        mode="w",
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {filename} - {funcName} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_file_handler.setFormatter(formatter)
    main_logging.setLevel(logging.DEBUG)
    main_logging.addHandler(logger_file_handler)

    main_logging.info("Log start")

    output_video_path = os.path.join(config.get_config["output"]["path"], "videos")
    os.makedirs(output_video_path, exist_ok=True)

    # Create folders to store the video and the stats
    os.makedirs(os.path.join(config.get_config["output"]["path"], "stats"), exist_ok=True)
    logging.info(f'Stats folder created at {os.path.join(config.get_config["output"]["path"], "stats")}')

    video_paths = glob.glob(
        os.path.join(config.get_config["input"]["path"], f"*.{config.get_config['input']['extension']}"))
    logging.info(f"List of videos to process: {video_paths}")

    if args.render_video_only:
        with parallel_backend("loky", verbose=100):
            Parallel(n_jobs=config.get_config["multiprocess"]["simultaneous_video_processes"])(
                (delayed(render_video)(video,
                                       os.path.join(config.get_config["output"]["path"], "stats"),
                                       config.get_config["input"]["path"],
                                       output_video_path,
                                       config) for video in config.get_config["output"]["render_videos"])
            )

    if args.interpolate_existing_tracks:
        with parallel_backend("loky", verbose=100):
            Parallel(n_jobs=config.get_config["multiprocess"]["simultaneous_video_processes"])(
                (delayed(interpolate_coordinates)(video,
                                                  os.path.join(config.get_config["output"]["path"], "stats"),
                                                  config.get_config["output"]["coordinates_columns"],
                                                  "linear",
                                                  config.get_config["output"]["interpolate"]["max_distance"]) for video in config.get_config["output"]["render_videos"])
            )

    if args.denormalized_existing_tracks:
        postfix = ""
        if args.interpolate_existing_tracks:
            postfix = "_interpolated"

        with parallel_backend("loky", verbose=100):
            Parallel(n_jobs=config.get_config["multiprocess"]["simultaneous_video_processes"])(
                (delayed(denormalize_coordinates)(video + postfix,
                                                  os.path.join(config.get_config["output"]["path"], "stats"),
                                                  config.get_config["output"]["coordinates_columns"],
                                                  "linear") for video in config.get_config["output"]["render_videos"])
            )

    if not (args.render_video_only or
            args.denormalized_existing_tracks or
            args.interpolate_existing_tracks) and args.track:
        # Use joblib to process videos, creating objects on demand
        with parallel_backend("loky", verbose=100):
            Parallel(n_jobs=config.get_config["multiprocess"]["simultaneous_video_processes"])(
                (delayed(create_job)(video_path, config, logging) for video_path in video_paths)
            )
        logging.info("All videos processed.")

    logger_file_handler.close()
    del main_logging


if __name__ == "__main__":
    main()
