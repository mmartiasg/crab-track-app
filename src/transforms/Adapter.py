class BaseYoloAdapter(object):
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


class YoloAdapter(BaseYoloAdapter):
    def __init__(self, video_name, tracker_name, dataset, column_names):
        super(YoloAdapter, self).__init__(video_name, tracker_name, dataset)
        self.column_names = column_names

    def create_record(self):
        record = {"dataset": self.dataset,
                  "tracker": self.tracker_name,
                  "video": self.video_name,
                  "frame": self.frame_index,
                  "inference_time[ms]": None,
                  "confidence": None,
                  "class_index": None
                  }

        record.update(dict(zip(self.column_names, [None, None, None, None])))

        return record

    def transform_record(self, bbox, results):
        record = self.create_record()

        try:
            (x1, y1, x2, y2) = bbox.xyxyn.squeeze().cpu().numpy()
            conf = bbox.conf.cpu().numpy()[0]
            class_index = int(bbox.cls.cpu().numpy()[0])
        except AttributeError:
            raise ValueError("Invalid bounding box format. Expected Decord/Numpy-compatible object.")

        record.update(dict(zip(self.column_names, [x1, y1, x2, y2])))

        record.update({
            "inference_time[ms]": results.speed["inference"],
            "confidence": conf,
            "class_index": class_index
        })

        return record


class DefaultAdapter(BaseYoloAdapter):
    def create_record(self):
        pass

    def transform_record(self, bbox, results):
        return results
