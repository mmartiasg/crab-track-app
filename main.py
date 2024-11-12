#!/usr/bin/env python3
import argparse
import os
from src.tracking.tracker import track_object, track_object_v2
from src.utils.constants import Config
from joblib import Parallel, delayed, parallel_backend
import glob
import multiprocessing as mpt
from src.tracking.videoRender import render_video
import logging


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

    output_video_path = None
    if config.get_config["output"]["export_videos"]:
        output_video_path = os.path.join(config.get_config["output"]["path"], "videos")
        os.makedirs(output_video_path, exist_ok=True)
        logging.info(f"Created folder to output videos in {output_video_path}")

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
                                     output_video_path=output_video_path,
                                     # keep the name of the video
                                     video_name=video_path.split("/")[-1].split(".")[0],
                                     tracker_name=f'version_{config.get_config["model"]["path"].split("/")[1]}',
                                     confidence_threshold=config.get_config["model"]["conf_threshold"],
                                     nms_threshold=config.get_config["model"]["nms_threshold"],
                                     device=config.get_config["model"]["device"],
                                     model_weights=config.get_config["model"]["path"],
                                     disable_progress_bar=config.get_config["output"]["disable_progress_bar"])
            for video_path in video_paths
        )

        # Note: This is here to wait for all the process to finish.
        # This place is where the post process are applied such as:
        # - Calculating statistics or
        # - Videos with the boundary box and traveled path
        for r in res:
            main_logging.info(f"Video: {r['input_video_path']} finished.")
            render_video(**r)

        logger_file_handler.close()
        del main_logging


if __name__ == "__main__":
    main()
