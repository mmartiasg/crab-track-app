import numpy as np
import pandas as pd


class CallbackInterpolateCoordinates:
    def __init__(self, coordinates_columns, method):
        self.coordinates_columns = coordinates_columns
        self.method = method

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        interpolated_coordinates = (coordinates_df[self.coordinates_columns]
                                    .interpolate(method=self.method)
                                    .ffill()
                                    .bfill()
                                    )

        coordinates_df[self.coordinates_columns] = interpolated_coordinates

        return coordinates_df


class CallbackDenormalizeCoordinates:
    def __init__(self, coordinates_columns, method, image_size):
        self.coordinates_columns = coordinates_columns
        self.image_size = image_size
        self.method = method

    def scale_matrix(self):
        return np.array([self.image_size[0],
                        self.image_size[1],
                        self.image_size[0],
                        self.image_size[1]]
                        )

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        coordinates_df[self.coordinates_columns] *= self.scale_matrix()
        coordinates_df[self.coordinates_columns] = coordinates_df[self.coordinates_columns].astype(int)

        return coordinates_df


class CallbackSaveToDisk:
    def __init__(self, file_path):
        self.file_path = file_path

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        coordinates_df.to_csv(self.file_path, index=False)

        return coordinates_df
