import logging
import pandas as pd
from ultralytics import YOLO
import os
from src.dataloaders.video_loader import VideoFramesGenerator
import torchvision
from src.transforms.Adapter import YoloAdapter
import multiprocessing as mpt


#TODO: Refactor this into a class tracker with methods set-up, track_step, track_videp and several hooks to call callbacks
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
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor()
    ])

    # TODO: move to config
    BATCH_SIZE = 64
    loader = VideoFramesGenerator(video_path=input_video_path,
                                  transform=None if use_yolo_tracker else video_frame_transform,
                                  batch_size=BATCH_SIZE,
                                  num_threads=mpt.cpu_count()//8)

    tracker_logging.debug(f"Video {video_name} loaded with frames: {loader.__len__()}")

    # Instance model
    model = YOLO(model_weights, task="detect", verbose=False)

    tracker_stats = []

    # TODO: move dataset name to config
    data_adapter = YoloAdapter(video_name=video_name, tracker_name=tracker_name, dataset="ICMAN3OCT2022")

    for batch in loader:
        # TODO: Refactor this in 2 tracker classes!
        # TODO: add option in confing to select tracker.
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

        records = data_adapter(batch_results)
        tracker_stats.extend(records)

        tracker_logging.debug(f"""
                    Frames processed: {data_adapter.frame_index} |
                    Frames with prediction over ({confidence_threshold}): {data_adapter.frames_with_measurement} |
                    Frames without prediction ({confidence_threshold}): {data_adapter.frames_without_measurement} |
                    Process time total: {round(data_adapter.inference_time_batch, 4)} ms, 
                    per frame: {round(data_adapter.inference_time_sample, 4)} ms
                    """)

    tracker_logging.info(f"Finished tracking {video_name}")

    # TODO: move this to a Checkpoint Callback that will save the file after each epoch.
    stats_df = None
    try:
        stats_df = pd.DataFrame([r[0] for r in tracker_stats])
    except Exception as e:
        tracker_logging.critical("Building dataframe", exc_info=True)
    # TODO: Maybe code a hooks where the callbacks can be called on after/before epoch or before start the whole process

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
