#!/usr/bin/env python3
import argparse
import os
from src.tracking.tracker import track_object
from src.utils.constants import Config
from joblib import Parallel, delayed
import glob
import multiprocessing as mpt
from tqdm.auto import tqdm
from src.utils.ProgressBarJoblib import tqdm_joblib


def main():
    parser = argparse.ArgumentParser(description="Configurations for the script")
    parser.add_argument("--config_path", help="Specify the config path", default=None)
    args = parser.parse_args()

    if args.config_path is not None:
        config = Config(args.config_path)
    else:
        raise (Exception("No config file specified"))

    if config.get_config["input"]["path"] is None:
        raise Exception("No path input path specified")

    output_video_path = ""
    if config.get_config["output"]["export_videos"]:
        output_video_path = os.path.join(config.get_config["output"]["path"], "videos")
        os.makedirs(output_video_path, exist_ok=True)

    # Create folders to store the video and the stats
    os.makedirs(os.path.join(config.get_config["output"]["path"], "stats"), exist_ok=True)

    video_paths = glob.glob(
        os.path.join(config.get_config["input"]["path"], f"*.{config.get_config['input']['extension']}"))

    with tqdm_joblib(tqdm(desc="Tracking", total=len(video_paths))):
        Parallel(n_jobs=mpt.cpu_count(), verbose=12)(
            delayed(track_object)(input_video_path=video_path,
                                  output_video_path=output_video_path,
                                  stats_path=os.path.join(config.get_config["output"]["path"], "stats"),
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


if __name__ == "__main__":
    main()
