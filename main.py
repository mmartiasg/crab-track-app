#!/usr/bin/env python3
import argparse
import os
from src.tracking.tracker import track_object_v2
from src.utils.constants import Config
from joblib import Parallel, delayed, parallel_backend
import glob
import multiprocessing as mpt
import logging
from src.callbacks.compose import ComposeCallback
from src.callbacks.post_processing import CallbackDenormalizeCoordinates, CallbackInterpolateCoordinates, \
    CallbackSaveToDisk
from src.callbacks.video_render import CallbackRenderVideo


def main():
    parser = argparse.ArgumentParser(description="Configurations for the script")
    parser.add_argument("--config_path", help="Specify the config path", default=None)
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
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
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

    with parallel_backend("loky", verbose=100):
        # TODO: use res to pruduce stats at the end or draw the path traveled in a video.
        res = Parallel(n_jobs=mpt.cpu_count() // 4, return_as="generator_unordered")(
            delayed(track_object_v2)(input_video_path=video_path,
                                     out_path=config.get_config["output"]["path"],
                                     # keep the name of the video
                                     video_name=video_path.split("/")[-1].split(".")[0],
                                     tracker_name=f'version_{config.get_config["model"]["path"].split("/")[1]}',
                                     confidence_threshold=config.get_config["model"]["conf_threshold"],
                                     nms_threshold=config.get_config["model"]["nms_threshold"],
                                     device=config.get_config["model"]["device"],
                                     model_weights=config.get_config["model"]["path"]
                                     )
            for video_path in video_paths
        )

        coordinates_columns = ["pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"]

        for r in res:
            callback_list = [
                CallbackInterpolateCoordinates(
                    coordinates_columns=coordinates_columns,
                    method="linear"),
                CallbackDenormalizeCoordinates(
                    coordinates_columns=coordinates_columns,
                    image_size=(config.get_config["input"]["resolution"]["width"],
                                config.get_config["input"]["resolution"]["height"]),
                    method="xyxy"),
                CallbackSaveToDisk(file_path=os.path.join(config.get_config["output"]["path"],
                                                          "stats",
                                                          r["video_name"] + "_post_processed.csv"))
            ]

            if r["video_name"] == "1_crop" or r["video_name"] == "12_crop.mp4":
                render_callback = CallbackRenderVideo(
                    output_video_path=os.path.join(config.get_config["output"]["path"],
                                                   "videos",
                                                   r["video_name"] + ".mp4"),
                    input_video_path=os.path.join(config.get_config["input"]["path"],
                                                  r["video_name"] + ".mp4"),
                    coordinate_columns=["pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"],
                    bbox_color=(0, 0, 255))
                callback_list.append(render_callback)

            compose_callback = ComposeCallback(callback_list)
            save_raw_data_callback = CallbackSaveToDisk(file_path=os.path.join(config.get_config["output"]["path"],
                                                                               "stats",
                                                                               r["video_name"] + ".csv"))
            callbacks = [compose_callback, save_raw_data_callback]

            for callback in callbacks:
                callback(r["coordinates"])
                main_logging.info(f"Video: {r['video_name']} finished.")

            del callbacks

        logger_file_handler.close()
        del main_logging


if __name__ == "__main__":
    main()
