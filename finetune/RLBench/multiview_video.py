# gbw___
"""Side-channel RLBench four-camera video writer."""

import os
import subprocess
import tempfile

import cv2
import numpy as np


CAMERAS = (
    ("front_rgb", "front", "_cam_front"),
    ("left_shoulder_rgb", "left_shoulder", "_cam_over_shoulder_left"),
    ("right_shoulder_rgb", "right_shoulder", "_cam_over_shoulder_right"),
    ("wrist_rgb", "wrist", "_cam_wrist"),
)


def _to_hwc_rgb(observation, key):
    image = np.asarray(observation[key])
    if image.ndim != 3:
        raise ValueError(f"{key} must be a 3D image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    elif image.shape[-1] != 3:
        raise ValueError(f"{key} must have three RGB channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"{key} must be uint8, got {image.dtype}")
    return image


def _label(image, name):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (125, 22), (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        name,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return labeled


def make_multiview_frame(observation):
    """Build a labeled RGB 2x2 frame without changing the observation."""
    images = [
        _label(_to_hwc_rgb(observation, key), name)
        for key, name, _ in CAMERAS
    ]
    height, width, channels = images[0].shape
    if channels != 3 or any(image.shape != (height, width, channels) for image in images):
        raise ValueError("All RGB camera images must have the same HWC shape")
    top = np.concatenate(images[:2], axis=1)
    bottom = np.concatenate(images[2:], axis=1)
    return np.concatenate((top, bottom), axis=0)


def _run_ffmpeg(command):
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr.strip()) from error


def _write_singleview_video(frames, output_path, fps=25, frame_dir=None):
    if not frames:
        raise ValueError("Cannot save a multiview video without rollout frames")

    height, width, _ = frames[0].shape
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".multiview_frames_", dir=frame_dir or os.path.dirname(output_path)
    ) as image_dir:
        for index, frame in enumerate(frames, start=1):
            frame_path = os.path.join(image_dir, f"{index:06d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                os.path.join(image_dir, "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                output_path,
            ]
        )

    return {"frames": len(frames), "width": width, "height": height, "fps": fps}


def _combine_singleview_videos(singleview_paths, output_path):
    top_filter = "[0:v][1:v]hstack=inputs=2[top]"
    bottom_filter = "[2:v][3:v]hstack=inputs=2[bottom]"
    final_filter = "[top][bottom]vstack=inputs=2[out]"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *sum((["-i", path] for path in singleview_paths), []),
            "-filter_complex",
            ";".join((top_filter, bottom_filter, final_filter)),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
    )


class MultiviewRecorder:
    """Capture physical cameras on every simulator step as a side channel."""

    def __init__(self, environment, resolution=256):
        self._environment = environment
        self._previous_callback = getattr(
            environment._task._scene, "_step_callback", None
        )
        self._resolution = [resolution, resolution]
        self._active = False
        self._camera_frames = {}

    def begin_episode(self):
        self._camera_frames = {key: [] for key, _, _ in CAMERAS}
        self._active = True

    def _capture_observation(self):
        scene = self._environment._task._scene
        observation = {}
        for key, _, camera_attribute in CAMERAS:
            camera = getattr(scene, camera_attribute)
            original_resolution = camera.get_resolution()
            try:
                if original_resolution != self._resolution:
                    camera.set_resolution(self._resolution)
                camera.handle_explicitly()
                image = camera.capture_rgb()
            finally:
                if original_resolution != self._resolution:
                    camera.set_resolution(original_resolution)
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            observation[key] = image
        return observation

    def _append_current_frame(self):
        observation = self._capture_observation()
        for key, name, _ in CAMERAS:
            self._camera_frames[key].append(
                _label(_to_hwc_rgb(observation, key), name)
            )

    def callback(self):
        if self._previous_callback is not None:
            self._previous_callback()
        if self._active:
            self._append_current_frame()

    def append_current_frame(self):
        """Match the single-view recorder's episode-end final frame."""
        if self._active:
            self._append_current_frame()

    def save_episode(self, output_path, fps=25):
        self._active = False
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        singleview_dir = os.path.join(
            os.path.dirname(output_path),
            "singleviews",
            os.path.splitext(os.path.basename(output_path))[0],
        )
        os.makedirs(singleview_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".multiview_singleviews_", dir=os.path.dirname(output_path)
        ) as temp_dir:
            singleview_paths = []
            metadata = None
            for key, _, _ in CAMERAS:
                singleview_path = os.path.join(singleview_dir, f"{key}.mp4")
                metadata = _write_singleview_video(
                    self._camera_frames[key],
                    singleview_path,
                    fps=fps,
                    frame_dir=temp_dir,
                )
                singleview_paths.append(singleview_path)
            _combine_singleview_videos(singleview_paths, output_path)
        return {
            "frames": metadata["frames"],
            "width": metadata["width"] * 2,
            "height": metadata["height"] * 2,
            "fps": metadata["fps"],
        }
#____
