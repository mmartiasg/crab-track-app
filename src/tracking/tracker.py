import logging
import pandas as pd
from ultralytics import YOLO
import os
from src.dataloaders.video_loader import VideoFramesGenerator
import torchvision


def track_object_v2(input_video_path,
                    out_path,
                    video_name,
                    tracker_name,
                    model_weights,
                    device="cpu",
                    confidence_threshold=0.8,
                    nms_threshold=0.5,
                    use_yolo_tracker=False):
    # Set up logger
    tracker_logging = logging.getLogger(__name__)
    logger_file_handler = logging.FileHandler(
        os.path.join(out_path, "logs", f"tracker_video_{video_name}.log"),
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

    tracker_logging.info(f"Start tracking {video_name}")

    video_frame_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor()
    ])

    BATCH_SIZE = 64
    loader = VideoFramesGenerator(video_path=input_video_path,
                                  transform=None if use_yolo_tracker else video_frame_transform,
                                  batch_size=BATCH_SIZE,
                                  num_threads=2)

    tracker_logging.debug(f"Video {video_name} loaded with frames: {loader.__len__()}")

    # Instance model
    model = YOLO(model_weights, task="detect", verbose=False)

    tracker_stats = []

    frame_index = 0
    frames_with_measurement = 0
    frames_without_measurement = 0
    for batch in loader:

        if use_yolo_tracker:
            # Track works with raw images
            # not sure how to feed it using a dataloader.
            batch_results = model.track(batch,
                                        persist=True,
                                        conf=confidence_threshold,
                                        iou=nms_threshold,
                                        verbose=False,
                                        device=device,
                                        stream=True)
        else:
            batch_results = model.predict(batch,
                                          conf=confidence_threshold,
                                          iou=nms_threshold,
                                          verbose=False,
                                          device=device,
                                          stream=True
                                          )
        # TODO: move this logic to an object responsible to build the record.
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
                    frames_with_measurement += 1
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
                frames_without_measurement += 1
                tracker_stats.append(record)
            frame_index += 1

        tracker_logging.debug(f"""
                    Frames processed: {frame_index} |
                    Frames with prediction over ({confidence_threshold}): {frames_with_measurement} |
                    Frames without prediction ({confidence_threshold}): {frames_without_measurement} |
                    Process time total: {round(results.speed["inference"], 4)},
                    per frame: {round(results.speed["inference"] / BATCH_SIZE, 4)} in ms
                    """)

    tracker_logging.info(f"Finished tracking {video_name}")

    # TODO: move this to a Checkpoint Callback that will save the file after each epoch.
    stats_df = None
    try:
        stats_df = pd.DataFrame(tracker_stats)
    except Exception as e:
        tracker_logging.critical("Building dataframe", exc_info=True)

    tracker_logging.info("Free resources allocated")
    for handler in tracker_logging.handlers:
        handler.close()
        tracker_logging.removeHandler(handler)

    del logger_file_handler
    del tracker_logging
    del loader
    del model

    return {
        "video_name": video_name,
        "coordinates": stats_df
    }
