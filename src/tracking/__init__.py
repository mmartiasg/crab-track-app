from src.tracking.yolo import TrackerByDetection, TrackerByByteTrack

TRACKER_CLASSES = {
    "detection": TrackerByDetection,
    "byte_track": TrackerByByteTrack
}