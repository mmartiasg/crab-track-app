#!/usr/bin/env python3
import argparse
import os
from tracking.tracker_yolov8 import track_object
from utils.constants import Config
from joblib import Parallel, delayed, parallel_backend
import glob


# with parallel_backend("loky", require="sharedmem"):
#     res = Parallel(n_jobs=8, return_as="generator_unordered")(
#         delayed(tag_new_samples)(os.path.join(VIDEOS_DIR, f"{video_id}.mp4"),
#                                  model_path=MODEL_WEIGHT_PATH,
#                                  tracker_name=MODEL_NAME + "_" + MODEL_VERSION,
#                                  threshold=threshold,
#                                  image_dir=IMAGE_DIR,
#                                  label_dir=LABEL_DIR,
#                                  video_id=video_id,
#                                  frames_skip=frames_skip) for video_id in TRAIN_VIDEOS_ID_LIST
#     )
#     print(res)
#     total_new_samples_added = np.sum([samples for samples in res])

def main():
    parser = argparse.ArgumentParser(description="Configurations for the script")
    parser.add_argument("--config_path", help="Specify the config file", default=None)
    args = parser.parse_args()

    if args.config_path is not None:
        config = Config(args.config_path)
    else:
        raise (Exception("No config file specified"))

    # Create folders to store the video and the stats
    os.makedirs(os.path.join(config.get_value("RESULTS_PATH"), "videos"), exist_ok=True)
    os.makedirs(os.path.join(config.get_value("RESULTS_PATH"), "stats"), exist_ok=True)

    for video_path in glob.glob(os.path.join(config.get_value("VIDEO_FOLDER"), f"*.{config.get_value('VIDEO_EXTENSION')}")):
        track_object(input_video_path=video_path,
                     output_video_path=os.path.join(config.get_value("RESULTS_PATH"), "videos"),
                     stats_path=os.path.join(config.get_value("RESULTS_PATH"), "stats"),
                     # keep the name of the video
                     video_name=video_path.split("/")[-1].split(".")[0],
                     tracker_name="YoloV8",
                     confidence=config.get_value("CONFIDENCE"),
                     device=config.get_value("DEVICE"),
                     model_weights=config.get_value("MODEL_WEIGHTS"))


if __name__ == "__main__":
    main()
