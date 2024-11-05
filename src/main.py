from tracking.tracker_yolov8 import track_object
import os


if __name__ == "__main__":
    lines = []

    WEIGHTS = "/home/matias/workspace/crab_track/yolo/exported_models/0.2.0/tracking.onnx"
    video_number = "8"
    INPUT_VIDEO_PATH = os.path.join("/home/matias/workspace/datasets/ICMAN-30-Octubre-2022", f"{video_number}.mp4")
    OUTPUT_VIDEO_PATH = f"{video_number}.mp4"

    track_object(input_video_path=INPUT_VIDEO_PATH,
                 output_video_path=OUTPUT_VIDEO_PATH,
                 video_name=str(video_number),
                 tracker_name="YoloV8",
                 model_weights=WEIGHTS)
