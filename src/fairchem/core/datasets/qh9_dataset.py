
import argparse
import json
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import lmdb
import numpy as np
import torch
from omegaconf import OmegaConf

from fairchem.core.common.registry import registry
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms

# Default units
Angstrom = 1.0
eV = 1.0

# Length unit (Angstrom in pyscf)
ANG2BOHR = 1.8897261258369282     # Angstrom to Bohr conversion
BOHR2ANG = 0.5291772105638411     # Bohr to Angstrom conversion

# Energy unit (Eh in pyscf)
HA2eV    = 27.211396641308        # Hartree to eV conversion
HA2meV   = HA2eV * 1000           # Hartree to meV conversion
eV2HA    = 0.03674932247495664    # eV to Hartree
meV2HA   = eV2HA / 1000           # meV to Hartree

KCALPM2eV  = 0.04336410390059322   # kcal/mol to eV conversion
KCALPM2meV = KCALPM2eV * 1000      # kcal/mol to meV conversion
HA2KCALPM  = 627.5094738898777     # Hartree to kcal/mol
KCALPM2HA  = 0.001593601438080425  # kcal/mol to Hartree

# Force unit (Eh/Bohr in pyscf)
HA_BOHR_2_KCALPM_ANG = HA2KCALPM / BOHR2ANG        # Hartree/Bohr to kcal/mol/Angstrom
KCALPM_ANG_2_HA_BOHR = 1.0 / HA_BOHR_2_KCALPM_ANG  # kcal/mol/Angstrom to Hartree/Bohr
HA_BOHR_2_meV_ANG    = HA2meV / BOHR2ANG           # Hartree/Bohr to meV/Angstrom
meV_ANG_2_HA_BOHR    = 1.0 / HA_BOHR_2_meV_ANG     # meV/Angstrom to Hartree/Bohr
HA_BOHR_2_HA_ANG     = 1.0 / BOHR2ANG              # Hartree/Bohr to Hartree/Angstrom
HA_ANG_2_HA_BOHR     = 1.0 / ANG2BOHR     

@registry.register_dataset("qh9")
class QH9AtomicDataset(BaseDataset):
    """
    Flexible dataset loader for QH9 LMDB shards.

    Supports processed dataset directories emitted by ``qh9_datasets_shard``. The
    loader can point either to the ``processed`` folder directly or to its parent
    project directory and will automatically locate ``index.json``, ``lmdbs/`` and the
    appropriate ``processed_*.json`` split file.
    """

    def __init__(
        self,
        lmdb_path: Optional[Union[str, Mapping[str, Any]]] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        processed_dir: Optional[str] = None,
        split: Optional[str] = None,
        split_filename: Optional[str] = None,
        split_type: Optional[str] = None,
        index_filename: str = "index.json",
        max_samples: Optional[int] = None,
        transform=None,
        debug: bool = False,
        debug_interval: int = 1000,
        **_: Any,
    ):
        resolved_config = self._prepare_config(
            config=config,
            lmdb_path=lmdb_path,
            processed_dir=processed_dir,
            split=split,
            split_filename=split_filename,
            split_type=split_type,
            index_filename=index_filename,
            max_samples=max_samples,
            transform=transform,
            debug=debug,
            debug_interval=debug_interval,
        )
        resolved_config = self._normalize_wrapped_config(resolved_config)

        super().__init__(resolved_config)

        self.transform = self.config.get("transform", None)
        self.data_transforms = DataTransforms(self.config.get("transforms", {}) or {})
        self.key_mapping = self.config.get("key_mapping", None)
        self.dataset_name = self.config.get("dataset_name", "qh9")
        self.dataset_names = [self.dataset_name]
        self.debug = bool(self.config.get("debug", False))
        self.debug_interval = max(int(self.config.get("debug_interval", 1000)), 1)

        self._envs: Dict[int, lmdb.Environment] = {}
        self._initialize_shard_key_cache()

        processed_dir_cfg = self._resolve_processed_dir(
            explicit=self._get_config_path("processed_dir"),
            lmdb_hint=self._get_config_path("lmdb_path"),
        )
        src_hint = processed_dir_cfg or self._get_config_path("src")

        if (not self.paths) and src_hint is not None:
            self.paths = [Path(src_hint)]

        if processed_dir_cfg is None:
            raise ValueError(
                "You must provide processed_dir (or a path from which it can be inferred) to QH9AtomicDataset."
            )

        self._init_multi_mode(processed_dir_cfg)

        self.num_samples = self.length
        self._natoms_cache: Dict[int, int] = {}

    def __len__(self):
        return self.length

    @staticmethod
    def _normalize_wrapped_config(
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(config, dict):
            config = OmegaConf.to_container(config, resolve=True)
        dataset_configs = config.pop("dataset_configs", None)
        if not dataset_configs:
            return config

        if isinstance(dataset_configs, Mapping):
            inner = next(iter(dataset_configs.values()))
        else:
            inner = dataset_configs[0]

        if not isinstance(inner, dict):
            inner = OmegaConf.to_container(inner, resolve=True)

        normalized = {k: v for k, v in inner.items()}
        for key, value in config.items():
            if key not in {"dataset_configs", "combined_dataset_config"}:
                normalized[key] = value

        splits_cfg = normalized.pop("splits", None)
        if splits_cfg:
            if isinstance(splits_cfg, Mapping):
                default_split = next(iter(splits_cfg))
                normalized.setdefault("split", default_split)
                selected = splits_cfg[normalized["split"]]
            else:
                normalized.setdefault("split", splits_cfg[0])
                selected = splits_cfg[normalized["split"]]

            if not isinstance(selected, dict):
                selected = OmegaConf.to_container(selected, resolve=True)

            for key, value in selected.items():
                normalized[key] = value

        return normalized

    def _init_multi_mode(self, processed_dir: str) -> None:
        split = self.config.get("split")
        if split is None:
            raise ValueError(
                "When using processed_dir you must provide which split ('train', 'val', 'test') to load."
            )

        self.processed_dir = os.path.abspath(processed_dir)
        if not self._is_valid_processed_dir(self.processed_dir):
            raise FileNotFoundError(
                f"Processed directory {self.processed_dir} is missing 'lmdbs' or 'index.json'."
            )

        self.split_name = split
        self.split_path = self._resolve_split_path(
            split_filename=self.config.get("split_filename"),
            split_type_hint=self.config.get("split_type"),
        )
        index_filename = self.config.get("index_filename", "index.json")
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
        split_indices: List[int] = list(split_info[split])
        max_samples = self.config.get("max_samples")
        if max_samples is not None:
            split_indices = split_indices[: max_samples]

        with open(self.index_path, "r") as f:
            index_info = json.load(f)["index"]

        self.sample_mappings = []
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
        self._envs.clear()
        self._initialize_shard_key_cache()
        self._debug(
            f"Initialized multi-shard dataset with {self.length} samples "
            f"from processed_dir={processed_dir}"
        )

    @staticmethod
    def _prepare_config(
        *,
        config: Optional[Mapping[str, Any]],
        lmdb_path: Optional[Union[str, Mapping[str, Any]]],
        processed_dir: Optional[str],
        split: Optional[str],
        split_filename: Optional[str],
        split_type: Optional[str],
        index_filename: str,
        max_samples: Optional[int],
        transform,
        debug: bool,
        debug_interval: int,
    ) -> Mapping[str, Any]:
        def legacy_args_supplied() -> bool:
            legacy_values = (
                processed_dir,
                split,
                split_filename,
                split_type,
                max_samples,
                transform,
            )
            return any(value is not None for value in legacy_values) or debug or (
                debug_interval != 1000
            ) or (index_filename != "index.json")

        if config is not None:
            if any(
                arg is not None
                for arg in (
                    lmdb_path if not isinstance(lmdb_path, Mapping) else None,
                    processed_dir,
                    split,
                    split_filename,
                    split_type,
                    max_samples,
                    transform,
                )
            ) or debug or (debug_interval != 1000) or (index_filename != "index.json"):
                raise ValueError(
                    "When providing a config mapping, legacy keyword arguments "
                    "such as lmdb_path or split may not be specified."
                )
            return config
        if isinstance(lmdb_path, Mapping):
            if legacy_args_supplied():
                raise ValueError(
                    "When passing a configuration mapping positionally, no legacy "
                    "keyword arguments may be supplied."
                )
            return lmdb_path

        resolved_config: Dict[str, Any] = {}
        if lmdb_path is not None:
            resolved_config["lmdb_path"] = lmdb_path
        if processed_dir is not None:
            resolved_config["processed_dir"] = processed_dir
        if split is not None:
            resolved_config["split"] = split
        if split_filename is not None:
            resolved_config["split_filename"] = split_filename
        if split_type is not None:
            resolved_config["split_type"] = split_type
        if index_filename != "index.json":
            resolved_config["index_filename"] = index_filename
        if max_samples is not None:
            resolved_config["max_samples"] = max_samples
        if transform is not None:
            resolved_config["transform"] = transform
        if debug:
            resolved_config["debug"] = debug
        if debug_interval != 1000:
            resolved_config["debug_interval"] = debug_interval

        return resolved_config

    def _get_config_path(self, key: str) -> Optional[str]:
        value = self.config.get(key)
        if value is None:
            return None
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, (str, bytes)):
            return str(value)
        if isinstance(value, Sequence):
            sequence_values = list(value)
            if len(sequence_values) == 0:
                return None
            if len(sequence_values) > 1:
                raise ValueError(
                    f"Expected a single entry for '{key}', but received {sequence_values}."
                )
            elem = sequence_values[0]
            if isinstance(elem, os.PathLike):
                return os.fspath(elem)
            return str(elem)
        return str(value)

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

    def _resolve_processed_dir(
        self, *, explicit: Optional[str], lmdb_hint: Optional[str]
    ) -> Optional[str]:
        """
        Determine which processed directory to use.

        Preference order:
            1. Explicit processed_dir from the config/constructor.
            2. Inferred directory derived from lmdb_path (for backwards compatibility).
        """
        if explicit:
            return explicit

        if lmdb_hint:
            inferred = self._maybe_infer_processed_dir(lmdb_hint)
            if inferred is None:
                raise ValueError(
                    "Could not infer processed_dir from lmdb_path. Please point "
                    "processed_dir directly to a directory containing lmdb shards."
                )
            return inferred

        return None

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
        # Make the graph fully connected (all pairs of atoms, including self-loops if needed)
        # Remove self-loops from the fully-connected graph
        n = natoms.item()
        row = torch.arange(n).repeat(n)
        col = torch.arange(n).repeat_interleave(n)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        cell_offsets = torch.empty((edge_index.shape[1], 3), dtype=torch.float)
        assert cell_offsets.shape[0] == edge_index.shape[1]
        nedges = torch.tensor([edge_index.shape[1]], dtype=torch.long)

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
            energy=energy * HA2meV,
            forces=forces * HA_BOHR_2_meV_ANG,
            sid=[idx_for_sid],
        )

        data.dataset_name = self.dataset_name
        data.dataset = getattr(data, "dataset_name", "qh9")
        data = self.data_transforms(data)
        if self.key_mapping is not None:
            data = rename_data_object_keys(data, self.key_mapping)
        if self.transform:
            data = self.transform(data)
        return data

    def metadata_hasattr(self, attr) -> bool:
        if attr == "natoms":
            return True
        return super().metadata_hasattr(attr)

    def get_metadata(self, attr, idx):
        metadata = super().get_metadata(attr, idx)
        if metadata is not None:
            return metadata
        if attr != "natoms":
            return None

        if isinstance(idx, slice):
            indices = list(range(*idx.indices(self.length)))
        elif isinstance(idx, np.ndarray):
            indices = idx.tolist()
        elif isinstance(idx, list):
            indices = self._flatten_index_list(idx)
        else:
            return self._get_natoms_for_index(idx)

        logging.info(
            "[QH9Dataset] Resolving natoms for %d indices (attr=%s)",
            len(indices),
            attr,
        )
        return np.array([self._get_natoms_for_index(i) for i in indices], dtype=np.int64)

    def export_metadata(
        self,
        output_path: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        overwrite: bool = False,
        report_interval: int = 1000,
        num_workers: int = 0,
    ) -> str:
        """
        Materialize metadata.npz containing at least natoms for every sample.

        Args:
            output_path: Optional explicit file path. Defaults to dataset's metadata path.
            overwrite: Allow overwriting an existing metadata file.
            report_interval: Print progress (when debug enabled) every this many samples.
            num_workers: Number of worker processes to use. <=1 falls back to serial mode.

        Returns:
            Absolute path to the written metadata file.
        """

        from tqdm import tqdm

        if self.length == 0:
            raise ValueError("Cannot export metadata for an empty dataset.")

        target_path = (
            Path(output_path).expanduser()
            if output_path is not None
            else self._default_metadata_path()
        )
        target_path = target_path.resolve()
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Metadata file already exists at {target_path}. "
                "Pass overwrite=True to replace it."
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        natoms = np.zeros(self.length, dtype=np.int32)

        print(f"Exporting metadata to {target_path}")

        if num_workers is None:
            num_workers = 0
        num_workers = int(max(num_workers, 0))
        print(f"[QH9Dataset] metadata workers: {num_workers}")

        if num_workers <= 1:
            from tqdm import tqdm

            iterator = tqdm(range(self.length), desc="Exporting metadata", unit="sample")
            self._populate_metadata_serial(natoms, iterator, report_interval)
        else:
            self._populate_metadata_parallel(natoms, num_workers)

        np.savez_compressed(target_path, natoms=natoms)
        self._debug(f"Wrote metadata.npz with {self.length} entries to {target_path}")
        return str(target_path)

    def _populate_metadata_serial(
        self,
        natoms: np.ndarray,
        iterator,
        report_interval: int,
    ) -> None:
        logging.info(
            "[QH9Dataset] Serial metadata pass for %d samples", self.length
        )
        shard_txns: Dict[int, lmdb.Transaction] = {}
        try:
            for idx in iterator:
                natoms[idx] = self._natoms_multi_mode(idx, shard_txns)
                if (
                    self.debug
                    and report_interval > 0
                    and (idx + 1) % report_interval == 0
                ):
                    self._debug(
                        f"Metadata progress: {idx + 1}/{self.length} samples processed."
                    )
        finally:
            for txn in shard_txns.values():
                txn.abort()

    def _populate_metadata_parallel(self, natoms: np.ndarray, num_workers: int) -> None:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        from tqdm import tqdm

        tasks = [
            (self.lmdb_dir, idx, mapping["shard_idx"], self._resolve_lmdb_key(mapping))
            for idx, mapping in enumerate(self.sample_mappings)
        ]
        logging.info(
            "[QH9Dataset] Parallel metadata pass for %d samples (%d workers)",
            self.length,
            num_workers,
        )

        worker_fn = _natoms_worker_entry
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for idx, value in tqdm(
                executor.map(worker_fn, tasks),
                total=self.length,
                desc="Metadata (parallel)",
                unit="sample",
            ):
                natoms[idx] = value
                self._natoms_cache[idx] = value

    def _default_metadata_path(self) -> Path:
        if self.config.get("metadata_path"):
            return Path(self.config["metadata_path"]).expanduser().resolve()
        if not self.paths:
            raise ValueError(
                "Cannot infer metadata output path. Please provide metadata_path."
            )

        base = Path(self.paths[0]).expanduser()
        base_exists = base.exists()
        if base_exists and base.is_file():
            target_dir = base.parent
        elif base_exists and base.is_dir():
            target_dir = base
        else:
            # Fall back to suffix heuristic if path does not yet exist.
            target_dir = base.parent if base.suffix else base

        target_dir = target_dir.resolve()

        filename = "metadata.npz"
        split_name = getattr(self, "split_name", None) or self.config.get("split")
        if split_name:
            filename = f"metadata_{split_name}.npz"

        return target_dir / filename

    def _get_natoms_for_index(self, idx: int) -> int:
        idx = int(idx)
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        if idx in self._natoms_cache:
            return self._natoms_cache[idx]

        data_dict, _ = self._fetch_raw_sample(idx)
        natoms = self._infer_natoms_from_record(data_dict)
        self._natoms_cache[idx] = natoms
        return natoms

    def _natoms_multi_mode(
        self, idx: int, shard_txns: Dict[int, lmdb.Transaction]
    ) -> int:
        mapping = self.sample_mappings[idx]
        shard_idx = mapping["shard_idx"]
        txn = shard_txns.get(shard_idx)
        if txn is None:
            env = self._get_or_open_env(shard_idx)
            txn = env.begin(write=False)
            shard_txns[shard_idx] = txn

        key_int = self._resolve_lmdb_key(mapping)
        key_bytes = self._encode_lmdb_key(key_int)
        value = txn.get(key_bytes)
        if value is None:
            raise ValueError(
                f"Key {key_int} not found in shard {shard_idx} "
                f"(global idx {mapping['global_idx']})"
            )
        natoms = self._natoms_from_serialized_value(value)
        self._natoms_cache[idx] = natoms
        return natoms

    @staticmethod
    def _natoms_from_serialized_value(value: bytes) -> int:
        data_dict = pickle.loads(value)
        return QH9AtomicDataset._infer_natoms_from_record(data_dict)

    @staticmethod
    def _flatten_index_list(idx_list: list) -> list[int]:
        flat: list[int] = []
        for item in idx_list:
            if isinstance(item, (list, tuple, np.ndarray)):
                flat.extend(int(i) for i in item)
            else:
                flat.append(int(item))
        return flat

    @staticmethod
    def _infer_natoms_from_record(data_dict: Dict[str, Any]) -> int:
        num_nodes = data_dict.get("num_nodes")
        if num_nodes is not None:
            try:
                return int(num_nodes)
            except (TypeError, ValueError):
                pass

        atoms_raw = data_dict.get("atoms")
        if atoms_raw is None:
            raise KeyError("Unable to infer natoms without 'atoms' field in record.")

        if isinstance(atoms_raw, bytes):
            natoms = len(np.frombuffer(atoms_raw, dtype=np.int32))
        else:
            natoms = len(atoms_raw)
        return int(natoms)

    def _fetch_raw_sample(self, idx: int) -> Tuple[Dict[str, Any], str]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        mapping = self.sample_mappings[idx]
        shard_idx = mapping["shard_idx"]
        key_int = self._resolve_lmdb_key(mapping)
        key_bytes = self._encode_lmdb_key(key_int)

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
        return data_dict, sid_value

    @staticmethod
    def _encode_lmdb_key(key_int: int) -> bytes:
        return int(key_int).to_bytes(length=4, byteorder="big")

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError

        if self.debug and idx % self.debug_interval == 0:
            self._debug(f"Fetching index {idx}/{self.length}")

        data_dict, sid_value = self._fetch_raw_sample(idx)
        return self._build_atomic_data(data_dict, sid_value)

    def close(self):
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


_WORKER_ENV_CACHE: dict[str, dict[int, lmdb.Environment]] = {}


def _natoms_worker_entry(payload: tuple[str, int, int, int]) -> tuple[int, int]:
    lmdb_dir, idx, shard_idx, key_int = payload
    envs = _WORKER_ENV_CACHE.setdefault(lmdb_dir, {})
    env = envs.get(shard_idx)
    shard_path = os.path.join(lmdb_dir, f"shard_{shard_idx:03d}.lmdb")

    if env is None:
        env = lmdb.open(
            shard_path,
            subdir=os.path.isdir(shard_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        envs[shard_idx] = env

    with env.begin(write=False) as txn:
        key_bytes = int(key_int).to_bytes(length=4, byteorder="big")
        value = txn.get(key_bytes)
        if value is None:
            raise ValueError(
                f"Key {key_int} not found in shard {shard_idx} ({shard_path})"
            )
        natoms = QH9AtomicDataset._natoms_from_serialized_value(value)
        return idx, natoms


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
        help="Which split to load (train/val/test). Use 'all' with --write_metadata to export every split.",
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
        default=None,
        help="Limit dataset to the first N samples (ignored when writing metadata).",
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
    parser.add_argument(
        "--write_metadata",
        action="store_true",
        help="Generate metadata.npz (natoms) for the requested split and exit.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Explicit metadata output path. When exporting multiple splits, provide a directory path.",
    )
    parser.add_argument(
        "--overwrite_metadata",
        action="store_true",
        help="Allow overwriting an existing metadata file.",
    )
    parser.add_argument(
        "--metadata_workers",
        type=int,
        default=0,
        help="Number of worker processes to use while exporting metadata.",
    )
    args = parser.parse_args()

    def _metadata_target_path(metadata_path: Optional[str], split_name: str, multi: bool):
        if metadata_path is None:
            return None
        base_path = Path(metadata_path).expanduser()
        if multi:
            if base_path.suffix:
                raise ValueError(
                    "When exporting multiple splits, metadata_path must be a directory path."
                )
            return str((base_path / f"metadata_{split_name}.npz").resolve())
        return str(base_path.resolve())

    if args.write_metadata:
        split_targets = (
            ["train", "val", "test"]
            if args.split.lower() == "all"
            else [args.split]
        )
        for split_name in split_targets:
            dataset = QH9AtomicDataset(
                processed_dir=args.processed_dir,
                split=split_name,
                split_filename=args.split_filename,
                split_type=args.split_type,
                max_samples=None,
                debug=args.debug,
                debug_interval=args.debug_interval,
            )
            print(
                f"[CLI] [{split_name}] Dataset length={len(dataset)} "
                f"(processed_dir={args.processed_dir})"
            )
            metadata_path = dataset.export_metadata(
                _metadata_target_path(
                    args.metadata_path, split_name, len(split_targets) > 1
                ),
                overwrite=args.overwrite_metadata,
                num_workers=args.metadata_workers,
            )
            print(f"[CLI] [{split_name}] Wrote metadata file to {metadata_path}")
        return

    if args.split.lower() == "all":
        raise ValueError(
            "Split 'all' is only supported when --write_metadata is specified."
        )

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

    preview_limit = args.max_samples if args.max_samples is not None else min(
        len(dataset), 16
    )
    for idx in range(min(len(dataset), preview_limit)):
        sample = dataset[idx]
        print(
            f"[CLI] idx={idx}, natoms={sample.natoms.item()}, "
            f"energy={getattr(sample, 'energy', None)}, "
            f"forces_shape={None if not hasattr(sample, 'forces') else sample.forces.shape}"
        )


if __name__ == "__main__":
    _cli()
