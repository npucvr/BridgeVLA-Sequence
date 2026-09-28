"""Stream raw RLBench episodes from packed task archives on shared NFS.

The authoritative raw data under ``/remote_databuffer`` is the per-task
archive tree, not an expanded small-file tree.  Compressed tar members cannot
be seeked cheaply, so this module walks each archive exactly once and yields
complete episodes in archive order.  The encoder never materializes millions
of tiny files on NFS.
"""

from __future__ import annotations

import io
import pickle
import tarfile
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

import numpy as np
from PIL import Image

from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor

CAMERA_FRONT = "front"
CAMERA_LS = "left_shoulder"
CAMERA_RS = "right_shoulder"
CAMERA_WRIST = "wrist"
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

IMAGE_RGB = "rgb"
IMAGE_DEPTH = "depth"
IMAGE_FORMAT = "%d.png"
LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_NUMBER_PICKLE = "variation_number.pkl"
VARIATION_DESCRIPTIONS_PICKLE = "variation_descriptions.pkl"
DEPTH_SCALE = 2**24 - 1

_TASK_ARCHIVE_SUFFIXES = (".tar.xz", ".tar.gz", ".tgz", ".tar")

# Files that make up one episode, relative to the episode directory.
_EPISODE_FILENAMES = {
    LOW_DIM_PICKLE,
    VARIATION_NUMBER_PICKLE,
    VARIATION_DESCRIPTIONS_PICKLE,
}


def _open_tar(path: Path) -> tarfile.TarFile:
    # Train archives are xz; eval archives named ``.tar.xz`` may be gzip.
    return tarfile.open(path, mode="r:*")


def _normalize_member(name: str) -> str:
    return name.lstrip("./")


def discover_task_archives(archives_root: Path) -> List[str]:
    root = Path(archives_root)
    if not root.is_dir():
        raise FileNotFoundError(f"archives root does not exist: {root}")
    tasks = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        for suffix in _TASK_ARCHIVE_SUFFIXES:
            if name.endswith(suffix):
                tasks.append(name[: -len(suffix)])
                break
    if not tasks:
        raise FileNotFoundError(f"no task archives under: {root}")
    return tasks


def archive_path_for_task(archives_root: Path, task: str) -> Path:
    root = Path(archives_root)
    for suffix in _TASK_ARCHIVE_SUFFIXES:
        candidate = root / f"{task}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no archive for task={task} under: {root}")


def _episode_id_from_member(name: str) -> Tuple[str, int, str] | None:
    """Return (episode_dir, episode_id, relative_path) or None."""
    parts = _normalize_member(name).split("/")
    try:
        episodes_index = parts.index("episodes")
    except ValueError:
        return None
    if episodes_index + 1 >= len(parts):
        return None
    episode_name = parts[episodes_index + 1]
    if not episode_name.startswith("episode"):
        return None
    suffix = episode_name[len("episode") :]
    if not suffix.isdigit():
        return None
    episode_dir = "/".join(parts[: episodes_index + 2])
    relative = "/".join(parts[episodes_index + 2 :])
    return episode_dir, int(suffix), relative


def _load_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image)


def _load_depth(data: bytes, near: float, far: float) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        depth = image_to_float_array(image, DEPTH_SCALE)
    return near + depth * (far - near)


def _attach_images(
    frame,
    *,
    camera: str,
    key: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
) -> None:
    setattr(frame, f"{key}_rgb", _load_rgb(rgb_bytes))
    near = frame.misc[f"{camera}_camera_near"]
    far = frame.misc[f"{camera}_camera_far"]
    depth = _load_depth(depth_bytes, near, far)
    setattr(frame, f"{key}_depth", depth)
    setattr(
        frame,
        f"{key}_point_cloud",
        VisionSensor.pointcloud_from_depth_and_camera_params(
            depth,
            frame.misc[f"{camera}_camera_extrinsics"],
            frame.misc[f"{camera}_camera_intrinsics"],
        ),
    )


def _build_demo(
    *,
    files: Mapping[str, bytes],
    episode_dir: str,
    episode_id: int,
):
    def need(name: str) -> bytes:
        key = f"{episode_dir}/{name}"
        if key not in files:
            raise FileNotFoundError(f"missing archive member: {key}")
        return files[key]

    obs = pickle.loads(need(LOW_DIM_PICKLE))
    try:
        obs.variation_number = pickle.loads(need(VARIATION_NUMBER_PICKLE))
    except FileNotFoundError:
        obs.variation_number = 0

    descriptions = pickle.loads(need(VARIATION_DESCRIPTIONS_PICKLE))
    if not descriptions:
        raise ValueError(f"episode has no variation description: {episode_dir}")
    description = str(descriptions[0])

    num_steps = len(obs)
    for i in range(num_steps):
        frame = obs[i]
        for camera, key in (
            (CAMERA_FRONT, "front"),
            (CAMERA_LS, "left_shoulder"),
            (CAMERA_RS, "right_shoulder"),
            (CAMERA_WRIST, "wrist"),
        ):
            _attach_images(
                frame,
                camera=camera,
                key=key,
                rgb_bytes=need(f"{camera}_{IMAGE_RGB}/{IMAGE_FORMAT % i}"),
                depth_bytes=need(f"{camera}_{IMAGE_DEPTH}/{IMAGE_FORMAT % i}"),
            )
    return obs, description


def iter_episodes_from_archive(
    archive_path: Path,
) -> Iterator[Tuple[int, object, str]]:
    """Yield ``(episode_id, demo, description)`` streaming the archive once."""
    archive_path = Path(archive_path)
    current_dir: str | None = None
    current_id: int | None = None
    files: Dict[str, bytes] = {}

    def flush():
        nonlocal current_dir, current_id, files
        if current_dir is None or current_id is None:
            return
        demo, description = _build_demo(
            files=files,
            episode_dir=current_dir,
            episode_id=current_id,
        )
        episode_id, demo_out, description_out = current_id, demo, description
        current_dir, current_id, files = None, None, {}
        return episode_id, demo_out, description_out

    with _open_tar(archive_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = _episode_id_from_member(member.name)
            if parsed is None:
                continue
            episode_dir, episode_id, relative = parsed
            if not relative:
                continue
            if current_dir is not None and episode_dir != current_dir:
                completed = flush()
                if completed is not None:
                    yield completed
            if current_dir is None:
                current_dir, current_id = episode_dir, episode_id
            handle = tar.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"archive member is not a file: {member.name}")
            with handle:
                files[f"{episode_dir}/{relative}"] = handle.read()

    completed = flush()
    if completed is not None:
        yield completed


def count_episodes_in_archive(archive_path: Path) -> int:
    count = 0
    with _open_tar(archive_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            parsed = _episode_id_from_member(member.name)
            if parsed is None:
                continue
            _episode_dir, episode_id, relative = parsed
            if relative == LOW_DIM_PICKLE:
                count += 1
    return count
