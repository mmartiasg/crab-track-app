import cv2
import os
import numpy as np
import pandas as pd


class CallbackRenderVideo:
    def __init__(self, output_video_path: str, input_video_path: str, coordinate_columns, bbox_color=(0, 0, 255)):
        self.cap = cv2.VideoCapture(input_video_path)

        # load video
        if not self.cap.isOpened():
            print('[ERROR] video file not loaded')

        self.coordinate_columns = coordinate_columns
        self.original_frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.bbox_color = bbox_color
        self.out = cv2.VideoWriter(
                os.path.join(output_video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                self.original_frame_rate,
                (self.width, self.height)
            )

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        coordinates = coordinates_df[self.coordinate_columns].values

        frame_index = 0
        path_points = np.zeros((self.frames_count, 2), dtype=int)

        ok, frame = self.cap.read()

        # Assuming 1 boundary box per frame.
        while ok:
            (x1, y1, x2, y2) = [int(v) for v in coordinates[frame_index]]

            # Draw the boundary box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.bbox_color, 2)

            # save the points to draw the path.
            path_points[frame_index, :] = [(x2 + x1) // 2, (y2 + y1) // 2]

            # Draw the path using previous points plus the new one for this frame.
            for i in range(frame_index):
                cv2.line(frame, tuple(path_points[i]), tuple(path_points[i + 1]), self.bbox_color, 5)

            # write frame to disk
            self.out.write(frame)

            # advance frame index
            frame_index += 1

            # get the next frame
            ok, frame = self.cap.read()

        # free resources
        self.cap.release()
        self.out.release()

        # to maintain the consistency among callbacks
        return coordinates_df
