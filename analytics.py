#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict


class RetailAnalytics:
    """Compute retail analytics from tracked-object metadata."""

    def __init__(self, metadata_file):
        with open(metadata_file) as f:
            self.frames = json.load(f)

        print(f"Loaded {len(self.frames)} frames")

    def count_entries_exits(self, tripwire_x=320):
        """Count objects crossing a virtual vertical tripwire."""

        entries = defaultdict(int)
        exits = defaultdict(int)
        object_sides = {}

        for frame in self.frames:
            for obj in frame.get("tracked_objects", []):
                obj_id = obj["id"]
                x = obj["x"]

                if obj_id not in object_sides:
                    object_sides[obj_id] = (
                        "left" if x < tripwire_x else "right"
                    )

                current_side = (
                    "left" if x < tripwire_x else "right"
                )
                previous_side = object_sides[obj_id]

                if current_side != previous_side:
                    if current_side == "right":
                        entries[obj_id] += 1
                    else:
                        exits[obj_id] += 1

                    object_sides[obj_id] = current_side

        return {
            "total_entries": sum(entries.values()),
            "total_exits": sum(exits.values()),
            "entries_by_object": dict(entries),
            "exits_by_object": dict(exits),
        }

    def compute_dwell_times(
        self,
        zone_x_min=100,
        zone_x_max=500,
        zone_y_min=100,
        zone_y_max=400,
    ):
        """Calculate time spent by tracked objects inside a zone."""

        dwell_times = defaultdict(
            lambda: {
                "entry_frame": None,
                "exit_frame": None,
                "frames_in_zone": 0,
            }
        )

        for frame_idx, frame in enumerate(self.frames):
            for obj in frame.get("tracked_objects", []):
                obj_id = obj["id"]
                x = obj["x"]
                y = obj["y"]

                in_zone = (
                    zone_x_min <= x <= zone_x_max
                    and zone_y_min <= y <= zone_y_max
                )

                if in_zone:
                    dwell_times[obj_id]["frames_in_zone"] += 1

                    if dwell_times[obj_id]["entry_frame"] is None:
                        dwell_times[obj_id]["entry_frame"] = frame_idx

                    dwell_times[obj_id]["exit_frame"] = frame_idx

        fps = 30.0
        result = {}

        for obj_id, data in dwell_times.items():
            if data["frames_in_zone"] > 0:
                dwell_seconds = data["frames_in_zone"] / fps

                if dwell_seconds > 30:
                    engagement = "active"
                elif dwell_seconds > 10:
                    engagement = "browsing"
                else:
                    engagement = "pass_through"

                result[obj_id] = {
                    "dwell_time_seconds": dwell_seconds,
                    "entry_frame": data["entry_frame"],
                    "exit_frame": data["exit_frame"],
                    "engagement": engagement,
                }

        return result

    def generate_heatmap(
        self,
        grid_width=100,
        grid_height=100,
        canvas_w=1920,
        canvas_h=1080,
    ):
        """Generate a grid-based occupancy heatmap."""

        grid = [
            [0 for _ in range(grid_width)]
            for _ in range(grid_height)
        ]

        for frame in self.frames:
            for obj in frame.get("tracked_objects", []):
                # Estimate foot position from bounding box.
                foot_x = int(
                    (obj["x"] + obj["width"] / 2)
                    / canvas_w
                    * grid_width
                )

                foot_y = int(
                    (obj["y"] + obj["height"])
                    / canvas_h
                    * grid_height
                )

                foot_x = min(
                    max(foot_x, 0),
                    grid_width - 1
                )

                foot_y = min(
                    max(foot_y, 0),
                    grid_height - 1
                )

                grid[foot_y][foot_x] += 1

        return grid

    def detect_queues(
        self,
        queue_x_min=300,
        queue_x_max=400,
        crowding_threshold=5,
    ):
        """Detect frames where a defined region becomes crowded."""

        queue_frames = []

        for frame_idx, frame in enumerate(self.frames):
            people_in_queue = sum(
                1
                for obj in frame.get("tracked_objects", [])
                if queue_x_min <= obj["x"] <= queue_x_max
            )

            if people_in_queue >= crowding_threshold:
                queue_frames.append({
                    "frame": frame_idx,
                    "timestamp": frame.get("timestamp"),
                    "people_count": people_in_queue,
                })

        return {
            "queue_detected_frames": len(queue_frames),
            "max_queue_length": max(
                (
                    frame["people_count"]
                    for frame in queue_frames
                ),
                default=0,
            ),
            "queue_events": queue_frames[:10],
        }

    def zone_engagement_analysis(self, zones=None):
        """Analyze visitor engagement across defined zones."""

        if zones is None:
            zones = {
                "checkout": (0, 200, 0, 1080),
                "produce": (200, 600, 0, 1080),
                "beverages": (600, 1200, 0, 1080),
                "dairy": (1200, 1920, 0, 1080),
            }

        zone_stats = defaultdict(
            lambda: {
                "frames": 0,
                "unique_people": set(),
            }
        )

        for frame in self.frames:
            for obj in frame.get("tracked_objects", []):
                x = obj["x"]
                y = obj["y"]

                for zone_name, (
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                ) in zones.items():

                    if (
                        x_min <= x <= x_max
                        and y_min <= y <= y_max
                    ):
                        zone_stats[zone_name]["frames"] += 1
                        zone_stats[zone_name][
                            "unique_people"
                        ].add(obj["id"])

        total_frames = len(self.frames)
        result = {}

        for zone, stats in zone_stats.items():
            result[zone] = {
                "foot_traffic_percentage": (
                    stats["frames"] / total_frames * 100
                    if total_frames > 0
                    else 0
                ),
                "unique_visitors": len(
                    stats["unique_people"]
                ),
                "total_frames_in_zone": stats["frames"],
            }

        return result


if __name__ == "__main__":
    metadata_file = "output/metadata_full_buffer.json"

    if not Path(metadata_file).exists():
        print(f"ERROR: {metadata_file} not found")
        print(
            "Run the DeepStream pipeline first:"
        )
        print(
            "  python3 deepstream_pipeline.py "
            "--source video.mp4"
        )
        raise SystemExit(1)

    analytics = RetailAnalytics(metadata_file)

    print("\n" + "=" * 60)
    print("ENTRY / EXIT ANALYTICS")
    print("=" * 60)

    counting = analytics.count_entries_exits()
    print(f"Total Entries: {counting['total_entries']}")
    print(f"Total Exits: {counting['total_exits']}")

    print("\n" + "=" * 60)
    print("DWELL TIME ANALYTICS")
    print("=" * 60)

    dwell = analytics.compute_dwell_times()

    if dwell:
        avg_dwell = (
            sum(
                data["dwell_time_seconds"]
                for data in dwell.values()
            )
            / len(dwell)
        )

        print(f"Average Dwell Time: {avg_dwell:.1f}s")

    print("\n" + "=" * 60)
    print("HEATMAP")
    print("=" * 60)

    heatmap = analytics.generate_heatmap()
    max_density = max(
        max(row) for row in heatmap
    )

    print(f"Max Grid Cell Density: {max_density}")

    print("\n" + "=" * 60)
    print("QUEUE ANALYSIS")
    print("=" * 60)

    queue = analytics.detect_queues()

    print(
        f"Frames with Queue: "
        f"{queue['queue_detected_frames']}"
    )
    print(
        f"Max Queue Length: "
        f"{queue['max_queue_length']} people"
    )

    print("\n" + "=" * 60)
    print("ZONE ENGAGEMENT")
    print("=" * 60)

    zones = analytics.zone_engagement_analysis()

    for zone, stats in zones.items():
        print(
            f"{zone}: "
            f"{stats['foot_traffic_percentage']:.1f}% "
            f"foot traffic, "
            f"{stats['unique_visitors']} visitors"
        )
