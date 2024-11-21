import cv2
import numpy as np
import pandas as pd


class CallbackRenderVideo:
    def __init__(self, output_video_path, input_video_path, coordinate_columns, bbox_color=(0, 0, 255)):
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
        self.frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # open video writer descriptor
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                              self.original_frame_rate,
                              (self.width, self.height)
                              )
        if not out.isOpened():
            raise Exception('[ERROR] cannot save the video')

        coordinates = coordinates_df[self.coordinate_columns].values

        frame_index = 0
        path_points = np.zeros((self.frames_count, 2), dtype=int)

        ok, frame = cap.read()

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
