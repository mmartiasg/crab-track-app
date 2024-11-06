#!/usr/bin/env python3
import argparse
import os
from src.tracking.tracker import track_object
from src.utils.constants import Config
from joblib import Parallel, delayed, parallel_backend
import glob
import multiprocessing as mpt


def main():
    parser = argparse.ArgumentParser(description="Configurations for the script")
    parser.add_argument("--config_path", help="Specify the config path", default=None)
    args = parser.parse_args()

    if args.config_path is not None:
        config = Config(args.config_path)
    else:
        raise (Exception("No config file specified"))

    # Create folders to store the video and the stats
    os.makedirs(os.path.join(config.get_config["output"]["path"], "videos"), exist_ok=True)
    os.makedirs(os.path.join(config.get_config["output"]["path"], "stats"), exist_ok=True)

    with parallel_backend("loky", require="sharedmem"):
        Parallel(n_jobs=mpt.cpu_count())(
            delayed(track_object)(input_video_path=video_path,
                                  output_video_path=os.path.join(config.get_config["output"]["path"], "videos"),
                                  stats_path=os.path.join(config.get_config["output"]["path"], "stats"),
                                  # keep the name of the video
                                  video_name=video_path.split("/")[-1].split(".")[0],
                                  tracker_name="YoloV8",
                                  confidence_threshold=config.get_config["model"]["conf_threshold"],
                                  nms_threshold=config.get_config["model"]["nms_threshold"],
                                  device=config.get_config["model"]["device"],
                                  model_weights=config.get_config["model"]["path"])
            for video_path in
            glob.glob(os.path.join(config.get_config["input"]["path"], f"*.{config.get_config['input']['extension']}"))
        )


if __name__ == "__main__":
    main()
