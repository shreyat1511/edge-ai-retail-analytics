#!/usr/bin/env python3
"""
DeepStream pipeline for retail video analytics.

Pipeline:
Video File → YOLO Detection → Object Tracking → JSON Metadata
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# DeepStream Python bindings
sys.path.insert(0, "/opt/nvidia/deepstream/deepstream-6.4/lib")

try:
    import pyds
except ImportError:
    print("ERROR: pyds module not found.")
    print(
        "Ensure DeepStream Python bindings are installed and "
        "LD_LIBRARY_PATH is configured correctly."
    )
    sys.exit(1)

from metadata_handler import MetadataExtractor, FrameMetadataBuffer


class DeepStreamRetailPipeline:
    """Build and run the DeepStream retail analytics pipeline."""

    def __init__(self, config_path="./config", output_dir="./output"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        Gst.init(None)

        self.pipeline = None
        self.bus = None
        self.loop = GLib.MainLoop()

        self.metadata_buffer = FrameMetadataBuffer(max_frames=300)

        self.frame_count = 0
        self.fps = 30.0
        self.start_time = None

    def build_pipeline(self, source_path):
        """Build the DeepStream pipeline for a recorded video file."""

        source_path = Path(source_path)

        if not source_path.exists():
            print(f"ERROR: Video file not found: {source_path}")
            return False

        pgie_config = self.config_path / "pgie_yolov8.txt"

        pipeline_str = f"""
            filesrc location="{source_path}" !
            qtdemux !
            h264parse !
            nvv4l2decoder !
            video/x-raw(memory:NVMM),format=NV12 !
            nvstreammux name=mux batch-size=1 batched-push-timeout=40000 !
            nvinfer config-file-path="{pgie_config}" name=pgie !
            nvtracker name=tracker !
            nvsink name=sink sync=false
        """

        try:
            self.pipeline = Gst.parse_launch(pipeline_str)

            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._on_message)

            tracker = self.pipeline.get_by_name("tracker")

            if tracker:
                src_pad = tracker.get_static_pad("src")

                if src_pad:
                    src_pad.add_probe(
                        Gst.PadProbeType.BUFFER,
                        self._probe_callback
                    )

                    print("[DeepStream] Metadata probe attached.")

            print("[DeepStream] Pipeline built successfully.")
            print(f"[DeepStream] Source: {source_path}")

            return True

        except Exception as exc:
            print(f"[ERROR] Failed to build pipeline: {exc}")
            return False

    def _probe_callback(self, pad, info, user_data=None):
        """Extract tracked-object metadata from each processed frame."""

        try:
            gst_buffer = info.get_buffer()

            if gst_buffer is None:
                return Gst.PadProbeReturn.OK

            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(
                hash(gst_buffer)
            )

            timestamp = (
                gst_buffer.pts / 1e9
                if gst_buffer.pts != Gst.CLOCK_TIME_NONE
                else self.frame_count / self.fps
            )

            tracked_objects = MetadataExtractor.extract_from_batch_meta(
                batch_meta,
                timestamp=timestamp
            )

            self.metadata_buffer.add_frame(
                tracked_objects,
                self.frame_count
            )

            if tracked_objects:
                self._write_frame_metadata(
                    tracked_objects,
                    self.frame_count
                )

            self.frame_count += 1

            if self.frame_count % 30 == 0:
                print(
                    f"[Progress] Frame {self.frame_count} | "
                    f"Time {timestamp:.2f}s | "
                    f"Objects: {len(tracked_objects)}"
                )

        except Exception as exc:
            print(f"[ERROR] Metadata extraction failed: {exc}")

        return Gst.PadProbeReturn.OK

    def _write_frame_metadata(self, tracked_objects, frame_id):
        """Write metadata for a processed frame."""

        output_file = self.output_dir / f"metadata_{frame_id:06d}.json"

        frame_data = {
            "frame_id": frame_id,
            "timestamp": tracked_objects[0].timestamp,
            "objects_count": len(tracked_objects),
            "tracked_objects": [
                obj.to_dict() for obj in tracked_objects
            ]
        }

        with open(output_file, "w") as file:
            json.dump(frame_data, file, indent=2)

    def _on_message(self, bus, message):
        """Handle GStreamer bus messages."""

        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()

            print(f"[ERROR] {error.message}")

            if debug:
                print(f"[DEBUG] {debug}")

            self.stop()

        elif message.type == Gst.MessageType.WARNING:
            warning, debug = message.parse_warning()
            print(f"[WARNING] {warning.message}")

        elif message.type == Gst.MessageType.EOS:
            print("[DeepStream] End of stream reached.")
            self.stop()

        return True

    def start(self):
        """Start the DeepStream pipeline."""

        if self.pipeline is None:
            print("ERROR: Pipeline has not been built.")
            return False

        result = self.pipeline.set_state(Gst.State.PLAYING)

        if result == Gst.StateChangeReturn.FAILURE:
            print("[ERROR] Failed to start pipeline.")
            return False

        self.start_time = datetime.now()

        print("[DeepStream] Pipeline started.")

        try:
            self.loop.run()

        except KeyboardInterrupt:
            print("\n[DeepStream] Interrupted by user.")
            self.stop()

        return True

    def stop(self):
        """Stop the pipeline and export accumulated metadata."""

        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        if self.loop.is_running():
            self.loop.quit()

        if self.metadata_buffer.frame_count > 0:
            output_file = (
                self.output_dir / "metadata_full_buffer.json"
            )

            self.metadata_buffer.export_json(str(output_file))

            print(
                f"[DeepStream] Metadata exported to {output_file}"
            )

        elapsed = (
            (datetime.now() - self.start_time).total_seconds()
            if self.start_time
            else 0
        )

        print(
            f"[DeepStream] Processed {self.frame_count} frames "
            f"in {elapsed:.2f}s."
        )


def main():
    parser = argparse.ArgumentParser(
        description="DeepStream retail video analytics pipeline"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to a recorded video file"
    )

    parser.add_argument(
        "--config",
        default="./config",
        help="Path to the configuration directory"
    )

    parser.add_argument(
        "--output",
        default="./output",
        help="Directory for generated metadata"
    )

    args = parser.parse_args()

    pipeline = DeepStreamRetailPipeline(
        config_path=args.config,
        output_dir=args.output
    )

    if pipeline.build_pipeline(args.source):
        pipeline.start()


if __name__ == "__main__":
    main()
