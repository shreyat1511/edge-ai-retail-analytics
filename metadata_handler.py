#!/usr/bin/env python3

import json
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class TrackedObject:
    """Represents a tracked object extracted from DeepStream metadata."""

    id: int
    x: float
    y: float
    width: float
    height: float
    timestamp: float
    confidence: float = None
    class_name: str = None

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict())


class MetadataExtractor:
    """Extract tracked-object metadata from a DeepStream batch."""

    @staticmethod
    def extract_from_batch_meta(batch_meta, timestamp):
        import pyds

        tracked_objects = []

        frame_list = batch_meta.frame_meta_list

        while frame_list is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
            except StopIteration:
                break

            obj_list = frame_meta.obj_meta_list

            while obj_list is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(obj_list.data)
                except StopIteration:
                    break

                tracked_id = obj_meta.object_id

                # Only include objects successfully assigned a tracking ID
                if tracked_id >= 0:
                    rect = obj_meta.rect_params

                    tracked_objects.append(
                        TrackedObject(
                            id=int(tracked_id),
                            x=float(rect.left),
                            y=float(rect.top),
                            width=float(rect.width),
                            height=float(rect.height),
                            timestamp=float(timestamp),
                            confidence=float(obj_meta.confidence),
                            class_name=str(obj_meta.obj_label),
                        )
                    )

                try:
                    obj_list = obj_list.next
                except StopIteration:
                    break

            try:
                frame_list = frame_list.next
            except StopIteration:
                break

        return tracked_objects

    @staticmethod
    def batch_to_json(batch_meta, timestamp):
        objects = MetadataExtractor.extract_from_batch_meta(
            batch_meta,
            timestamp
        )

        return json.dumps(
            [obj.to_dict() for obj in objects],
            indent=2
        )


class FrameMetadataBuffer:
    """Rolling buffer of frame-level tracking metadata."""

    def __init__(self, max_frames=300):
        self.max_frames = max_frames
        self.frames = deque(maxlen=max_frames)

    def add_frame(self, frame_id, timestamp, tracked_objects):
        self.frames.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "tracked_objects": [
                obj.to_dict()
                for obj in tracked_objects
            ],
        })

    def get_object_trajectory(self, object_id, lookback_frames=30):
        """Return recent position history for a tracked object."""

        trajectory = []

        recent_frames = list(self.frames)[-lookback_frames:]

        for frame in recent_frames:
            for obj in frame["tracked_objects"]:
                if obj["id"] == object_id:
                    trajectory.append({
                        "frame_id": frame["frame_id"],
                        "timestamp": frame["timestamp"],
                        "x": obj["x"],
                        "y": obj["y"],
                    })

        return trajectory

    def export_json(self, filepath):
        """Export the complete metadata buffer to JSON."""

        with open(filepath, "w") as f:
            json.dump(
                list(self.frames),
                f,
                indent=2
            )


if __name__ == "__main__":
    # Simple local test without requiring DeepStream.
    test_object = TrackedObject(
        id=1,
        x=100,
        y=200,
        width=50,
        height=120,
        timestamp=0.033,
        confidence=0.95,
        class_name="person",
    )

    print("Single object JSON:")
    print(test_object.to_json())

    buffer = FrameMetadataBuffer(max_frames=10)

    buffer.add_frame(
        frame_id=0,
        timestamp=0.033,
        tracked_objects=[test_object]
    )

    print("\nFrame metadata:")
    print(json.dumps(list(buffer.frames), indent=2))

    print("\nObject trajectory:")
    print(buffer.get_object_trajectory(1))
