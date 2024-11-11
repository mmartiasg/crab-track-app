import cv2
import logging
import random
import time
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO
import numpy as np
import os
import sys
from src.dataloaders.video_loader import VideoDataloader
from torch.utils.data import DataLoader
import torchvision
import multiprocessing as mpt

mpt.set_start_method('fork', force=True)


def track_object_v2(input_video_path,
                    out_path,
                    video_name,
                    tracker_name,
                    model_weights,
                    device="cpu",
                    confidence_threshold=0.8,
                    nms_threshold=0.5,
                    disable_progress_bar=None,
                    output_video_path=None):

    # Set up logger
    tracker_logging = logging.getLogger(__name__)
    logger_file_handler = logging.FileHandler(
                                os.path.join(out_path, "logs", f"tracker_video_{video_name}.log"),
                                mode="w",
                                encoding="utf-8"
                            )
    formatter = logging.Formatter(
                    "{asctime} - {levelname} - {message}",
                    style="{",
                    datefmt="%Y-%m-%d %H:%M",
                )
    logger_file_handler.setFormatter(formatter)
    tracker_logging.setLevel(logging.DEBUG)
    tracker_logging.addHandler(logger_file_handler)

    tracker_logging.info(f"Start tracking {video_name}")

    video_frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    loader = VideoDataloader(video_path=input_video_path,
                             transform=video_frame_transform)

    tracker_logging.debug(f"Video {video_name} loaded with frames: {loader.__len__()}")

    # Instance model
    model = YOLO(model_weights, task="detect", verbose=False)

    tracker_stats = []
    BATCH_SIZE = 128
    data_loader = DataLoader(loader, batch_size=BATCH_SIZE, shuffle=False, num_workers=mpt.cpu_count() // 4)

    frame_index = 0
    frames_with_meassurements = 0
    frames_without_meassurements = 0
    for batch in data_loader:
        # Track works with raw images
        # not sure how to feed it using a dataloader.
        # batch_results = model.track(batch,
        #                       # persist=True,
        #                       conf=confidence_threshold,
        #                       iou=nms_threshold,
        #                       verbose=False,
        #                       device=device,
        #                       imgsz=(640, 640))

        batch_results = model.predict(batch,
                                      conf=confidence_threshold,
                                      iou=nms_threshold,
                                      verbose=False,
                                      device=device
                                      )

        for results in batch_results:
            if len(results.boxes) > 0:
                for bbox in results.boxes:
                    (x1, y1, x2, y2) = bbox.xyxyn.squeeze().cpu().numpy()
                    record = {"dataset": "ICMAN3OCT2022",
                              "tracker": tracker_name,
                              "video": video_name,
                              "frame": frame_index,
                              "inference_time": results.speed["inference"] / BATCH_SIZE,
                              "pred_bbox_x1": x1,
                              "pred_bbox_y1": y1,
                              "pred_bbox_x2": x2,
                              "pred_bbox_y2": y2
                              }
                    tracker_stats.append(record)
                    frames_with_meassurements += 1
            else:
                record = {"dataset": "ICMAN3OCT2022",
                          "tracker": tracker_name,
                          "video": video_name,
                          "frame": frame_index,
                          "inference_time": results.speed["inference"] / BATCH_SIZE,
                          "pred_bbox_x1": None,
                          "pred_bbox_y1": None,
                          "pred_bbox_x2": None,
                          "pred_bbox_y2": None
                          }
                frames_without_meassurements += 1
                tracker_stats.append(record)
            frame_index += 1

        tracker_logging.debug(f"""
                    Frames processed: {frame_index} |
                    Frames with prediction over ({confidence_threshold}): {frames_with_meassurements} |
                    Frames without prediction ({confidence_threshold}): {frames_without_meassurements} |
                    Process time total: {round(results.speed["inference"], 4)},
                    per frame: {round(results.speed["inference"] / BATCH_SIZE, 4)} in ms
                    """)

    stats_df = None
    try:
        stats_df = pd.DataFrame(tracker_stats)
        stats_df.to_csv(os.path.join(out_path, "stats", f"{video_name}.csv"), index=False)
    except Exception as e:
        tracker_logging.critical("Saving dataframe", exc_info=True)

    tracker_logging.info(f"Finished tracking {video_name}")

    tracker_logging.info("Free resources allocated")

    # close logger file handlers
    for handler in tracker_logging.handlers:
        handler.close()
        tracker_logging.removeHandler(handler)

    del logger_file_handler
    del tracker_logging
    del loader
    del data_loader
    del model

    return {
        "input_video_path": input_video_path,
        "output_video_path": None,
        "coordinates": stats_df.values if stats_df is not None else None
    }


def track_object(input_video_path,
                 output_video_path,
                 stats_path,
                 video_name,
                 tracker_name,
                 model_weights,
                 device="cpu",
                 confidence_threshold=0.8,
                 nms_threshold=0.5,
                 disable_progress_bar=True):
    """
    This runs all trackers defined in the trackers_name_list for one video shows every frame in a window
    Saves the results in results, and the resulted video will be saved in this same directory with the same name.

    Parameters
    ----------
    :param disable_progress_bar: True or false to disable the progress bar.
    :param input_video_path: Path where the video is located.
    :param stats_path: Path where the stats file is located.
    :param video_name: Name of the video that will be used to create the output video and the result files.
    :param tracker_name: Name of the tracker that will be used to create the output video.
    :param model_weights: Path where the model weights are located.
    :param device: cpu, cuda or mps.
    :param nms_threshold: NMS threshold from 0 to 1.0.
    :param confidence_threshold: Confidence threshold from 0 to 1.0.
    :param output_video_path: Path where the output video is located.
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

    # TODO: Log to file
    # print(f'[INFO] video: {video_name} loaded and frame capture started')

    original_frame_rate = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_video_path is not None:
        out = cv2.VideoWriter(
            os.path.join(output_video_path, video_name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), original_frame_rate,
            (width, height)
        )

    # load video
    if out is not None and not out.isOpened():
        print(
            f'[INFO] Writer not initialized check the output path {os.path.join(output_video_path, video_name + ".mp4")}')

    # Instanciate model
    model = YOLO(model_weights, task="detect", verbose=False)

    # random generate a colour for bounding box
    random.seed(132)
    bbox_color = dict()

    bbox_color[tracker_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # loop through all frames of video file
    pbar = tqdm(total=frames_count,
                ncols=100,
                file=sys.stderr,
                disable=disable_progress_bar,
                desc=f"Video: {video_name}")
    frame_index = 0

    while True:
        ok, frame = video.read()

        if not ok:
            # TODO: log to file
            # print(f'[INFO] end of video file {video_name} reached')
            break

        start_time = time.time()

        results = model.track(frame,
                              persist=True,
                              conf=confidence_threshold,
                              iou=nms_threshold,
                              verbose=False,
                              device=device)

        duration = time.time() - start_time

        for bbox in results[0].boxes:
            (x1, y1, x2, y2) = bbox.xyxy.squeeze().cpu().numpy().astype(np.int32)

            if out is not None and out.isOpened():
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
        pbar.set_description(f"Video: {video_name}")
        pbar.update(1)

        if out is not None and out.isOpened():
            out.write(frame)

    stats_df = pd.DataFrame(tracker_stats)
    # interpolate frames based on the previous and next frames when missing.
    stats_df = stats_df[["pred_bbox_x1", "pred_bbox_y1", "pred_bbox_x2", "pred_bbox_y2"]].interpolate()
    stats_df.to_csv(os.path.join(stats_path, f"{video_name}.csv"))

    pbar.close()
    video.release()
    if out is not None and out.isOpened():
        out.release()

    del model

    return {"input_video_path": input_video_path,
            "output_video_path": output_video_path,
            "coordinates": stats_df.values}
