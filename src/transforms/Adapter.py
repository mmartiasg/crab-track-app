class BaseAdapter(object):
    def __init__(self, video_name, tracker_name, dataset):
        self.video_name = video_name
        self.tracker_name = tracker_name
        self.frame_index = 0
        self.frames_with_measurement = 0
        self.frames_without_measurement = 0
        self.dataset = dataset
        self.inference_time_batch = 0
        self.inference_time_sample = 0

    def create_record(self):
        raise NotImplementedError("Implement in subclass")

    def transform_record(self, bbox, results):
        raise NotImplementedError("Implement in subclass")

    def __call__(self, batch_results):
        self.inference_time_batch = 0
        frame_records = []

        for results in batch_results:

            record_boxes = [
                self.transform_record(bbox, results)
                for bbox in results.boxes
            ]

            # the inference is done per image which is one per frame
            # is not done per bbox detected
            self.inference_time_batch += results.speed["inference"]

            if len(results.boxes) > 0:
                self.frames_with_measurement += 1
            else:
                self.frames_without_measurement += 1
                record_boxes.append(self.create_record())

            self.frame_index += 1
            frame_records.append(record_boxes)

        num_frames = len(frame_records)
        self.inference_time_sample = self.inference_time_batch / num_frames if num_frames else 0

        return frame_records


class YoloAdapter(BaseAdapter):
    def create_record(self):
        record = {"dataset": self.dataset,
                  "tracker": self.tracker_name,
                  "video": self.video_name,
                  "frame": self.frame_index,
                  "x1": None,
                  "y1": None,
                  "x2": None,
                  "y2": None
                  }

        return record

    def transform_record(self, bbox, results):
        record = self.create_record()

        try:
            (x1, y1, x2, y2) = bbox.xyxyn.squeeze().cpu().numpy()
        except AttributeError:
            raise ValueError("Invalid bounding box format. Expected Decord/Numpy-compatible object.")

        record.update({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "inference_time[ms]": results.speed["inference"],
        })

        return record


class DefaultAdapter(BaseAdapter):
    def create_record(self):
        pass

    def transform_record(self, bbox, results):
        return results
