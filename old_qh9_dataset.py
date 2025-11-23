
import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from fairchem.core.datasets.atomic_data import AtomicData


class QH9AtomicDataset(Dataset):
    """
    Flexible dataset loader for QH9 LMDB shards.

    Two operating modes are supported:
        1. Single LMDB file (used for toy or custom data).
        2. Full processed dataset directories emitted by ``qh9_datasets_shard``. The
           loader can point either to the ``processed`` folder directly or to its
           parent project directory and will automatically locate ``index.json``,
           ``lmdbs/`` and the appropriate ``processed_*.json`` split file.
    """

    def __init__(
        self,
        lmdb_path: Optional[str] = None,
        *,
        processed_dir: Optional[str] = None,
        split: Optional[str] = None,
        split_filename: Optional[str] = None,
        split_type: Optional[str] = None,
        index_filename: str = "index.json",
        max_samples: Optional[int] = None,
        transform=None,
        debug: bool = False,
        debug_interval: int = 1000,
    ):
        self.transform = transform
        self.mode: str
        self.debug = debug
        self.debug_interval = max(debug_interval, 1)

        inferred_processed = self._maybe_infer_processed_dir(lmdb_path)
        if processed_dir is None and inferred_processed is not None:
            processed_dir = inferred_processed

        if processed_dir is not None:
            if split is None:
                raise ValueError(
                    "When using processed_dir you must provide which split ('train', 'val', 'test') to load."
                )

            self.mode = "multi"
            self.processed_dir = os.path.abspath(processed_dir)
            if not self._is_valid_processed_dir(self.processed_dir):
                raise FileNotFoundError(
                    f"Processed directory {self.processed_dir} is missing 'lmdbs' or 'index.json'."
                )

            self.split_name = split
            self.split_path = self._resolve_split_path(
                split_filename=split_filename,
                split_type_hint=split_type,
            )
            self.index_path = os.path.join(self.processed_dir, index_filename)
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(f"index file not found: {self.index_path}")
            self.lmdb_dir = os.path.join(self.processed_dir, "lmdbs")

            with open(self.split_path, "r") as f:
                split_info = json.load(f)
            if split not in split_info:
                raise ValueError(
                    f"Split '{split}' not found in {self.split_path}. "
                    f"Available: {list(split_info.keys())}"
                )
            split_indices: List[int] = split_info[split]
            if max_samples is not None:
                split_indices = split_indices[: max_samples]

            with open(self.index_path, "r") as f:
                index_info = json.load(f)["index"]

            self.sample_mappings: List[Dict[str, int]] = []
            for global_idx in split_indices:
                shard_idx, cur_idx, shard_data_idx = index_info[global_idx]
                if cur_idx != global_idx:
                    raise ValueError(
                        f"Index mismatch for global idx {global_idx}: got {cur_idx}"
                    )
                self.sample_mappings.append(
                    {
                        "shard_idx": int(shard_idx),
                        "shard_data_idx": int(shard_data_idx),
                        "global_idx": int(global_idx),
                    }
                )

            self.length = len(self.sample_mappings)
            self._envs: Dict[int, lmdb.Environment] = {}
            self._subdir_lookup: Dict[int, bool] = {}
            self._initialize_shard_key_cache()
            self._debug(
                f"Initialized multi-shard dataset with {self.length} samples "
                f"from processed_dir={processed_dir}"
            )

        elif lmdb_path is not None:
            self.mode = "single"
            self.lmdb_path = lmdb_path
            self.subdir = os.path.isdir(lmdb_path)
            self.env = lmdb.open(
                lmdb_path,
                subdir=self.subdir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            with self.env.begin(write=False) as txn:
                self.length = txn.stat()["entries"]

            self.keys = None
            if self.subdir:
                keys_path = os.path.join(lmdb_path, "keys_list.json")
                if os.path.exists(keys_path):
                    with open(keys_path, "r") as f:
                        data = json.load(f)
                    self.keys = [x[0] for x in data["keys_list"]]
                    if len(self.keys) != self.length:
                        print(
                            "Warning: keys_list length "
                            f"{len(self.keys)} != LMDB entries {self.length}"
                        )
            self._debug(
                f"Initialized single LMDB dataset with {self.length} entries "
                f"(path={lmdb_path})"
            )
        else:
            raise ValueError(
                "You must provide either lmdb_path or processed_dir to QH9AtomicDataset."
            )

    def __len__(self):
        return self.length

    @staticmethod
    def _is_valid_processed_dir(path: str) -> bool:
        return os.path.isdir(path) and os.path.isdir(
            os.path.join(path, "lmdbs")
        ) and os.path.isfile(os.path.join(path, "index.json"))

    def _maybe_infer_processed_dir(self, lmdb_path: Optional[str]) -> Optional[str]:
        if lmdb_path is None:
            return None

        abs_path = os.path.abspath(lmdb_path)
        if os.path.isdir(abs_path):
            if self._is_valid_processed_dir(abs_path):
                return abs_path
            candidate = os.path.join(abs_path, "processed")
            if self._is_valid_processed_dir(candidate):
                return candidate
        return None

    def _resolve_split_path(
        self,
        *,
        split_filename: Optional[str],
        split_type_hint: Optional[str],
    ) -> str:
        if split_filename is not None:
            split_path = (
                split_filename
                if os.path.isabs(split_filename)
                else os.path.join(self.processed_dir, split_filename)
            )
            if not os.path.exists(split_path):
                raise FileNotFoundError(f"Split file not found: {split_path}")
            return split_path

        available = [
            fname
            for fname in os.listdir(self.processed_dir)
            if fname.startswith("processed_") and fname.endswith(".json")
        ]
        if not available:
            raise FileNotFoundError(
                f"No processed split files found under {self.processed_dir}"
            )

        split_hint = (split_type_hint or "random").lower()
        preferred = [f for f in available if "ordered" not in f.lower()]

        def _match(candidates):
            return [
                f
                for f in candidates
                if split_hint in f.lower()
            ]

        matches = _match(preferred) or _match(available)
        if not matches:
            matches = preferred or available

        chosen = matches[0]
        return os.path.join(self.processed_dir, chosen)

    def _get_or_open_env(self, shard_idx: int) -> lmdb.Environment:
        if shard_idx in self._envs:
            return self._envs[shard_idx]

        shard_path = os.path.join(self.lmdb_dir, f"shard_{shard_idx:03d}.lmdb")
        subdir = os.path.isdir(shard_path)
        env = lmdb.open(
            shard_path,
            subdir=subdir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self._envs[shard_idx] = env
        self._subdir_lookup[shard_idx] = subdir
        self._debug(f"Opened LMDB env for shard {shard_idx}")
        return env

    @staticmethod
    def _to_tensor_from_blob(blob, dtype, reshape: Optional[Tuple[int, ...]] = None):
        if isinstance(blob, bytes):
            arr = np.frombuffer(blob, dtype=dtype).copy()
        else:
            arr = np.array(blob, dtype=dtype, copy=True)
        if reshape is not None:
            arr = arr.reshape(reshape)
        return torch.from_numpy(arr)

    def _build_atomic_data(self, data_dict, idx_for_sid: str) -> AtomicData:
        num_nodes = data_dict.get("num_nodes")

        pos_raw = data_dict["pos"]
        if isinstance(pos_raw, bytes):
            pos = self._to_tensor_from_blob(pos_raw, np.float64, (-1, 3)).float()
        else:
            pos = torch.from_numpy(np.array(pos_raw, dtype=np.float64)).float()

        atoms_raw = data_dict["atoms"]
        if isinstance(atoms_raw, bytes):
            atomic_numbers = (
                self._to_tensor_from_blob(atoms_raw, np.int32).long()
            )
        else:
            atomic_numbers = torch.from_numpy(np.array(atoms_raw, dtype=np.int32)).long()

        natoms = torch.tensor(
            [num_nodes if num_nodes is not None else atomic_numbers.shape[0]],
            dtype=torch.long,
        )

        forces_raw = data_dict.get("dft_forces")
        forces = None
        if forces_raw is not None:
            if isinstance(forces_raw, bytes):
                forces = (
                    self._to_tensor_from_blob(forces_raw, np.float64, (-1, 3))
                    .float()
                )
            else:
                forces = torch.from_numpy(np.array(forces_raw, dtype=np.float64)).float()
            if forces.shape != pos.shape:
                forces = forces.reshape(pos.shape)

        energy = data_dict.get("dft_energy")
        if energy is not None:
            energy = torch.tensor([energy], dtype=torch.float)

        cell = torch.zeros(1, 3, 3, dtype=torch.float)
        pbc = torch.zeros(1, 3, dtype=torch.bool)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        cell_offsets = torch.empty((0, 3), dtype=torch.float)
        nedges = torch.tensor([0], dtype=torch.long)

        data = AtomicData(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=torch.tensor([0], dtype=torch.long),
            spin=torch.tensor([0], dtype=torch.long),
            fixed=torch.zeros(natoms.item(), dtype=torch.long),
            tags=torch.zeros(natoms.item(), dtype=torch.long),
            energy=energy,
            forces=forces,
            sid=[idx_for_sid],
        )

        if self.transform:
            data = self.transform(data)
        return data

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        if self.debug and idx % self.debug_interval == 0:
            self._debug(f"Fetching index {idx}/{self.length}")

        if self.mode == "single":
            if self.keys is not None:
                real_key = self.keys[idx]
                key_bytes = int(real_key).to_bytes(length=4, byteorder="big")
            else:
                key_bytes = int(idx).to_bytes(length=4, byteorder="big")
            with self.env.begin(write=False) as txn:
                value = txn.get(key_bytes)
            if value is None:
                raise ValueError(
                    f"Key {idx} (bytes {key_bytes}) not found in LMDB {self.lmdb_path}"
                )
            data_dict = pickle.loads(value)
            return self._build_atomic_data(data_dict, str(data_dict.get("id", idx)))

        # multi mode
        mapping = self.sample_mappings[idx]
        shard_idx = mapping["shard_idx"]
        key_int = self._resolve_lmdb_key(mapping)
        key_bytes = int(key_int).to_bytes(length=4, byteorder="big")

        env = self._get_or_open_env(shard_idx)
        with env.begin(write=False) as txn:
            value = txn.get(key_bytes)

        if value is None:
            raise ValueError(
                f"Key {key_int} not found in shard {shard_idx} "
                f"(global idx {mapping['global_idx']})"
            )

        data_dict = pickle.loads(value)
        sid_value = str(data_dict.get("id", mapping["global_idx"]))
        return self._build_atomic_data(data_dict, sid_value)

    def close(self):
        if self.mode == "single":
            self.env.close()
        else:
            for env in self._envs.values():
                env.close()
            self._envs.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _debug(self, msg: str):
        if self.debug:
            print(f"[QH9Dataset] {msg}")

    def _initialize_shard_key_cache(self) -> None:
        """Prepare lazy cache for per-shard keys."""
        self._shard_keys: Dict[int, Optional[List[int]]] = {}

    def _resolve_lmdb_key(self, mapping: Dict[str, int]) -> int:
        shard_idx = mapping["shard_idx"]
        keys = self._get_shard_keys(shard_idx)
        if keys:
            shard_data_idx = mapping["shard_data_idx"]
            if shard_data_idx < len(keys):
                return keys[shard_data_idx]
        return mapping["global_idx"]

    def _get_shard_keys(self, shard_idx: int) -> Optional[List[int]]:
        cache = getattr(self, "_shard_keys", None)
        if cache is None:
            self._initialize_shard_key_cache()
            cache = self._shard_keys

        if shard_idx in cache:
            return cache[shard_idx]

        shard_path = os.path.join(self.lmdb_dir, f"shard_{shard_idx:03d}.lmdb")
        keys_path = os.path.join(shard_path, "keys_list.json")
        if not os.path.exists(keys_path):
            cache[shard_idx] = None
            return None

        try:
            with open(keys_path, "r") as f:
                raw = json.load(f).get("keys_list", [])
            keys = [
                int(item[0]) if isinstance(item, (list, tuple)) else int(item)
                for item in raw
            ]
            cache[shard_idx] = keys
            self._debug(
                f"Lazy-loaded {len(keys)} keys from {os.path.relpath(keys_path)}"
            )
            return keys
        except Exception as exc:
            self._debug(f"Failed to load keys_list for shard {shard_idx}: {exc}")
            cache[shard_idx] = None
            return None


def _cli():
    parser = argparse.ArgumentParser(description="Debug QH9AtomicDataset loading.")
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="/root/25DFT/QHFlow/dataset/QH9Stable_shard/processed",
        help="Path to processed QH9 directory (containing lmdbs/, index.json, etc.)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to load (train/val/test).",
    )
    parser.add_argument(
        "--split_filename",
        type=str,
        default="processed_QH9Stable_random_12.json",
        help="Split JSON filename if processed_dir has multiple options.",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default=None,
        help="Optional hint to choose split file (e.g., random, size_ood).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=16,
        help="Maximum number of samples to iterate for debugging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose dataset debugging output.",
    )
    parser.add_argument(
        "--debug_interval",
        type=int,
        default=1,
        help="How frequently (in samples) to print dataset progress when debug is enabled.",
    )
    args = parser.parse_args()

    dataset = QH9AtomicDataset(
        processed_dir=args.processed_dir,
        split=args.split,
        split_filename=args.split_filename,
        split_type=args.split_type,
        max_samples=args.max_samples,
        debug=args.debug,
        debug_interval=args.debug_interval,
    )
    print(
        f"[CLI] Dataset length={len(dataset)} "
        f"(processed_dir={args.processed_dir}, split={args.split})"
    )

    for idx in range(min(len(dataset), args.max_samples)):
        sample = dataset[idx]
        print(
            f"[CLI] idx={idx}, natoms={sample.natoms.item()}, "
            f"energy={getattr(sample, 'energy', None)}, "
            f"forces_shape={None if not hasattr(sample, 'forces') else sample.forces.shape}"
        )


if __name__ == "__main__":
    _cli()
