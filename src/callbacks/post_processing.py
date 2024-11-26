import numpy as np
import pandas as pd
from fsspec import Callback


class AbstractCallback(Callback):
    def __name__(self):
        return self.__class__.__name__


class CallbackInterpolateCoordinatesSingleObjectTracking(AbstractCallback):
    def __init__(self, coordinates_columns, method, max_distance, **kwargs):
        super().__init__(**kwargs)
        self.coordinates_columns = coordinates_columns
        self.method = method
        self.max_distance = max_distance

    # TODO: modify this to interpolate the same instance object in a multi object track environment.
    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:

        indices = coordinates_df.index[coordinates_df[self.coordinates_columns].notna().all(axis=1)]
        for i in range(len(indices) - 1):
            # Check if the distance between consecutive non-NaN points is within the limit
            if indices[i + 1] - indices[i] <= self.max_distance:
                # Interpolate between the points
                coordinates_df.loc[indices[i]: indices[i + 1], self.coordinates_columns] = coordinates_df.loc[indices[i]: indices[i + 1], self.coordinates_columns].interpolate()

            # Step 2: Fill remaining NaNs at the beginning
            if coordinates_df[self.coordinates_columns].isna().iloc[0].any():  # If NaNs exist in the first row
                coordinates_df[self.coordinates_columns] = coordinates_df[self.coordinates_columns].bfill()

            # Step 2: Fill remaining NaNs at the end
            if coordinates_df[self.coordinates_columns].isna().iloc[-1].any():  # If NaNs exist in the last row
                coordinates_df[self.coordinates_columns] = coordinates_df[self.coordinates_columns].ffill()

        return coordinates_df


class CallbackDenormalizeCoordinates(AbstractCallback):
    def __init__(self, coordinates_columns, method, image_size, **kwargs):
        super().__init__(**kwargs)
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
        placeholder = -1

        coordinates_df[self.coordinates_columns] *= self.scale_matrix()
        coordinates_df[self.coordinates_columns] = (coordinates_df[self.coordinates_columns]
                                                    .fillna(placeholder).astype(int).replace(placeholder, np.nan))

        return coordinates_df


class CallbackSaveToDisk(AbstractCallback):
    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path

    def __call__(self, coordinates_df: pd.DataFrame) -> pd.DataFrame:
        coordinates_df.to_csv(self.file_path, index=False)

        return coordinates_df
