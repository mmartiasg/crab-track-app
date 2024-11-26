import cv2
import numpy as np
import pandas as pd
from numpy._typing import NDArray

from src.callbacks.post_processing import AbstractCallback


class CallbackRenderVideoSingleObjectTracking(AbstractCallback):
    def __init__(self, output_video_path, input_video_path, coordinate_columns, bbox_color=(0, 0, 255), **kwargs):
        super().__init__(**kwargs)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.coordinate_columns = coordinate_columns
        self.bbox_color = bbox_color

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        # open video capture descriptor
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            raise Exception('[ERROR] video file not loaded')

        # video data
        self.original_frame_rate = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # CAP_PROP_FRAME_COUNT does an estimation and in some cases rounds up to one less frame
        self.frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

        # open video writer descriptor
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              self.original_frame_rate,
                              (self.width, self.height)
                              )
        if not out.isOpened():
            raise Exception('[ERROR] cannot save the video')

        coordinates = coordinates_df[self.coordinate_columns].values

        frame_index = 0
        path_points = np.ones((self.frames_count, 2), dtype=int) * -1

        ok, frame = cap.read()

        # Assuming 1 boundary box per frame.
        while ok:
            if coordinates_df.iloc[frame_index][self.coordinate_columns].notna().all():
                (x1, y1, x2, y2) = [int(v) for v in coordinates[frame_index]]

                # Draw the boundary box
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)

                # save the points to draw the path.
                path_points[frame_index, :] = [(x2 + x1) // 2, (y2 + y1) // 2]

            # Draw the path using previous points plus the new one for this frame.
            for i in range(frame_index):
                if ((path_points[i] >= 0).all() and
                        (path_points[i + 1] >= 0).all() and
                        euclidean_distance(path_points[i], path_points[i + 1]) < 50):
                    cv2.line(frame, tuple(path_points[i]), tuple(path_points[i + 1]), self.bbox_color, 5)

            # write frame to disk
            out.write(frame)

            # advance frame index
            frame_index += 1

            # get the next frame
            ok, frame = cap.read()

        # free resources
        cap.release()
        out.release()

        del cap
        del out

        # to maintain the consistency among callbacks
        return coordinates_df


def euclidean_distance(points_1: NDArray, points_2: NDArray) -> float:
    return ((points_1[0] - points_2[0]) ** 2 + (points_1[1] - points_2[1]) ** 2) ** 0.5
