"""NFS-efficient RLBench eval packs.

Official ``get_stored_demos`` expands camera PNGs even though eval reset only
needs ``low_dim_obs.pkl`` (``Demo.random_seed`` + observations), variation
number, and descriptions.  This pack stores just those metadata files — one
small archive per episode — and provides a loader with the same Demo
semantics.  Full images stay in ``original/archives`` and can be materialized
separately if a caller truly needs them.
"""

from __future__ import annotations

import io
import json
import pickle
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

FORMAT_NAME = "bridgevla_eval_pack"
FORMAT_VERSION = 1

LOW_DIM_PICKLE = "low_dim_obs.pkl"
VARIATION_NUMBER_PICKLE = "variation_number.pkl"
VARIATION_DESCRIPTIONS_PICKLE = "variation_descriptions.pkl"
_META_FILES = (
    LOW_DIM_PICKLE,
    VARIATION_NUMBER_PICKLE,
    VARIATION_DESCRIPTIONS_PICKLE,
)


def pack_paths(root: Path, task: str, episode_id: int) -> Path:
    return Path(root) / task / f"episode{int(episode_id)}.tar"


def write_episode_pack(
    *,
    root: Path,
    task: str,
    episode_id: int,
    files: Mapping[str, bytes],
) -> Path:
    """Write one compact episode pack from ``name -> bytes`` metadata files."""
    root = Path(root)
    task_dir = root / task
    task_dir.mkdir(parents=True, exist_ok=True)
    pack_path = pack_paths(root, task, episode_id)
    tmp_path = pack_path.with_suffix(pack_path.suffix + ".partial")
    episode_prefix = f"episode{int(episode_id)}"
    with tarfile.open(tmp_path, mode="w") as tar:
        for name in _META_FILES:
            if name not in files:
                raise KeyError(f"missing metadata file {name} for {task}#{episode_id}")
            data = files[name]
            info = tarfile.TarInfo(name=f"{episode_prefix}/{name}")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
    tmp_path.replace(pack_path)
    return pack_path


def write_manifest(root: Path, manifest: Mapping[str, object]) -> None:
    path = Path(root) / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()


def read_manifest(root: Path) -> dict:
    path = Path(root) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def list_tasks(root: Path) -> List[str]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"eval pack root does not exist: {root}")
    tasks = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not tasks:
        raise FileNotFoundError(f"no tasks under eval pack root: {root}")
    return tasks


def list_episode_ids(root: Path, task: str) -> List[int]:
    task_dir = Path(root) / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"missing eval pack task: {task_dir}")
    ids = []
    for path in task_dir.glob("episode*.tar"):
        suffix = path.stem[len("episode") :]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def _read_pack_files(pack_path: Path) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    with tarfile.open(pack_path, mode="r:") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name not in _META_FILES:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise FileNotFoundError(f"pack member is not a file: {member.name}")
            with handle:
                files[name] = handle.read()
    missing = [name for name in _META_FILES if name not in files]
    if missing:
        raise RuntimeError(f"pack {pack_path} missing {missing}")
    return files


def load_demo(root: Path, task: str, episode_id: int):
    """Return the same ``Demo`` object official ``get_stored_demos`` would."""
    pack_path = pack_paths(root, task, episode_id)
    if not pack_path.is_file():
        raise FileNotFoundError(f"missing eval episode pack: {pack_path}")
    files = _read_pack_files(pack_path)
    demo = pickle.loads(files[LOW_DIM_PICKLE])
    demo.variation_number = pickle.loads(files[VARIATION_NUMBER_PICKLE])
    descriptions = pickle.loads(files[VARIATION_DESCRIPTIONS_PICKLE])
    if not descriptions:
        descriptions = ["unknown task description"]
    for i in range(len(demo)):
        demo[i].misc["descriptions"] = list(descriptions)
    return demo


def load_descriptions(root: Path, task: str, episode_id: int) -> List[str]:
    pack_path = pack_paths(root, task, episode_id)
    if not pack_path.is_file():
        raise FileNotFoundError(f"missing eval episode pack: {pack_path}")
    files = _read_pack_files(pack_path)
    return list(pickle.loads(files[VARIATION_DESCRIPTIONS_PICKLE]))


def get_stored_demos(
    *,
    root: Path,
    task_name: str,
    amount: int,
    variation_number: int = -1,
    random_selection: bool = True,
    from_episode_number: int = 0,
    rng=None,
):
    """Drop-in behavioral match for RLBench ``get_stored_demos`` on packs.

    Image fields are left unset because eval ``reset_to_demo`` only needs
    ``Demo.random_seed``, variation number, and descriptions.  Official image
    loading is unchanged when using the expanded ``dataset_root`` tree.
    """
    import numpy as np

    ids = list_episode_ids(root, task_name)
    if variation_number != -1:
        filtered = []
        for episode_id in ids:
            demo = load_demo(root, task_name, episode_id)
            if int(demo.variation_number) == int(variation_number):
                filtered.append(episode_id)
        ids = filtered
    if not ids:
        raise RuntimeError(
            f"Can't find the demos for {task_name} at: {Path(root) / task_name}"
        )
    if amount == -1:
        amount = len(ids)
    if amount > len(ids):
        raise RuntimeError(
            f"You asked for {amount} examples, but only {len(ids)} were available."
        )
    if random_selection:
        chooser = rng if rng is not None else np.random
        selected = chooser.choice(ids, amount, replace=False)
        if np.isscalar(selected):
            selected = [int(selected)]
        else:
            selected = [int(x) for x in selected]
    else:
        selected = ids[from_episode_number : from_episode_number + amount]
    return [load_demo(root, task_name, episode_id) for episode_id in selected]
