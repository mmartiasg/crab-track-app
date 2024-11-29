import cv2
import numpy as np
import pandas as pd
from numpy._typing import NDArray

from src.callbacks.post_processing import AbstractCallback


class CallbackRenderVideoTracking(AbstractCallback):
    def __init__(self, output_video_path, input_video_path, coordinate_columns, bbox_color=(0, 0, 255), **kwargs):
        super().__init__(**kwargs)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.coordinate_columns = coordinate_columns
        self.bbox_color = bbox_color

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        # Open video capture descriptor
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise Exception('[ERROR] video file not loaded')

        # Video properties
        original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

        # Open video writer descriptor
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              original_frame_rate, (width, height))
        if not out.isOpened():
            raise Exception('[ERROR] cannot save the video')

        # Preprocess coordinates DataFrame
        # This is to treat every frame with their own bbox set for each object in screen.
        grouped_coordinates = coordinates_df.groupby("frame")[self.coordinate_columns].apply(np.array)

        # Path points storage
        path_points = np.full((frames_count, 2), -1, dtype=int)

        # Process video frames
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Get coordinates for the current frame (if any)
            # This returns a np array this is due to the .apply(np.array)
            coordinates = grouped_coordinates.get(frame_index, None)
            if coordinates is not None:
                for coordinate in coordinates:
                    if not np.isnan(coordinate).any():
                        # Extract coordinates and convert each position to int in one step!.
                        x1, y1, x2, y2 = map(int, coordinate)

                        # Draw the bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)

                        # Save path points
                        path_points[frame_index] = [(x1 + x2) // 2, (y1 + y2) // 2]

            # Draw the path using precomputed points
            valid_points = path_points[path_points[:, 0] >= 0]
            if len(valid_points) > 1:
                distances = np.linalg.norm(np.diff(valid_points, axis=0), axis=1)
                for i, dist in enumerate(distances):
                    if dist <= 100:  # Threshold for drawing paths
                        cv2.line(frame, tuple(valid_points[i]), tuple(valid_points[i + 1]), self.bbox_color, 5)

            # Write the frame to the output video
            out.write(frame)

            # Increment frame index
            frame_index += 1

        # Release resources
        cap.release()
        out.release()

        # Maintain consistency among callbacks
        return coordinates_df


def euclidean_distance(points_1: NDArray, points_2: NDArray) -> float:
    return np.linalg.norm(points_1 - points_2)
