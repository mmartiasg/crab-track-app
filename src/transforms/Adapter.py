class YoloAdapter(object):
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
        record = {"dataset": self.dataset,
                  "tracker": self.tracker_name,
                  "video": self.video_name,
                  "frame": self.frame_index,
                  "pred_bbox_x1": None,
                  "pred_bbox_y1": None,
                  "pred_bbox_x2": None,
                  "pred_bbox_y2": None
                  }

        return record

    def __call__(self, batch_results):
        self.inference_time_batch = 0
        frame_records = []

        for results in batch_results:
            record_boxes = []

            for bbox in results.boxes:
                record = self.create_record()
                (x1, y1, x2, y2) = bbox.xyxyn.squeeze().cpu().numpy()
                record["inference_time[ms]"] = results.speed["inference"]
                record["pred_bbox_x1"] = x1
                record["pred_bbox_y1"] = y1
                record["pred_bbox_x2"] = x2
                record["pred_bbox_y2"] = y2
                record_boxes.append(record)

            # the inference is done per image which is one per frame
            # is not done per bbox detected
            self.inference_time_batch += results.speed["inference"]
            self.frame_index += 1

            if len(record_boxes) > 0:
                self.frames_without_measurement += 1
            else:
                self.frames_without_measurement += 1
                record_boxes.append(self.create_record())

            frame_records.append(record_boxes)

        self.inference_time_sample = self.inference_time_batch / len(frame_records)

        return frame_records
