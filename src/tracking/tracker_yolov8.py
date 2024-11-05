import cv2
import os
import random
import time
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO
import numpy as np

DATASET_BASE_DIR = '../../../datasets/crab-dataset.v1i.yolov8'
IMAGE_DIR = os.path.join(DATASET_BASE_DIR, "all_data", 'images')
LABEL_DIR = os.path.join(DATASET_BASE_DIR, "all_data", 'labels')


def track_object(input_video_path, output_video_path, video_name, tracker_name, model_weights):
    """
    This runs all trackers defined in the trackers_name_list for one video shows every frame in a window
    Saves the results in results, and the resulted video will be saved in this same directory with the name video_name+_bbox_preds.mp4

    Parameters
    ----------
    input_video_path: Path where the video is located
    video_name: Name of the video that will be used to create the output video and the result files.
    trackers_name_list: Tracker list that will be used to create the output video and the result files.
    ground_truth: Ground truth of the video that will be used in the notebook evaluation_single_video.ipynb to evaluate the trackers over the video
    """

    video = cv2.VideoCapture(input_video_path)

    # initialize dictionaries
    tracker_stats = []

    # load video
    if not video.isOpened():
        print('[ERROR] video file not loaded')
    # capture first frame
    ok, frame = video.read()
    if not ok:
        print('[ERROR] no frame captured')

    print(f'[INFO] video: {video_name} loaded and frame capture started')

    original_frame_rate = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), original_frame_rate, (width, height)
    )

    # Load model
    model = YOLO(model_weights, task="detect")

    # random generate a colour for bounding box
    random.seed(132)
    bbox_color = {}

    bbox_color[tracker_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # loop through all frames of video file
    pbar = tqdm(total=frames_count)
    frame_index = 0

    while True:
        ok, frame = video.read()

        if not ok:
            print('[INFO] end of video file reached')
            break

        start_time = time.time()
        results = model.track(frame, persist=True, conf=0.7, verbose=False, device="cuda")
        duration = time.time() - start_time

        for bbox in results[0].boxes:
            (x1, y1, x2, y2) = bbox.xyxy.squeeze().cpu().numpy().astype(np.int32)

            # use predicted bounding box coordinates to draw a rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color[tracker_name], 3)
            cv2.putText(frame, f"{tracker_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        bbox_color[tracker_name], 2)

            record = {"dataset": "ICMAN3OCT2022",
                      "tracker": tracker_name,
                      "video": video_name,
                      "frame": frame_index,
                      "inference_time": duration,
                      "pred_bbox_x1": x1,
                      "pred_bbox_y1": y1,
                      "pred_bbox_x2": x2,
                      "pred_bbox_y2": y2
                      }

            tracker_stats.append(record)

        frame_index += 1
        pbar.update(1)
        out.write(frame)

    stats_df = pd.DataFrame(tracker_stats)

    # interpolate frames based on the previous and next frames when missing.
    stats_df = stats_df[["pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"]].interpolate()

    stats_df.to_csv(f"../results/stats/stats_{video_name}.csv")

    pbar.close()
    video.release()
    out.release()
    del model

