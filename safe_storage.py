import json
import os
import pickle
from datetime import datetime
from pathlib import Path

DATABASE_VERSION = 1
DEFAULT_DATABASE_PATH = "face_database.json"
LEGACY_DATABASE_SUFFIXES = {".pkl", ".pickle"}
_WINDOWS_RESERVED_CHARS = {":", "*", "?", '"', "<", ">", "|"}


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to process the face database.") from exc
    return np


def sanitize_filename_component(value, fallback="item", max_length=80):
    """Sanitize a single filename component while keeping readable Unicode names."""
    text = str(value).strip()
    if not text:
        return fallback

    sanitized_chars = []
    for char in text:
        if char in {"/", "\\"} or char in _WINDOWS_RESERVED_CHARS or ord(char) < 32:
            sanitized_chars.append("_")
        else:
            sanitized_chars.append(char)

    sanitized = "".join(sanitized_chars).strip(" .")
    if sanitized in {"", ".", ".."}:
        sanitized = fallback

    return sanitized[:max_length]


def safe_output_filename(original_name, prefix="", fallback="file"):
    """Build a filesystem-safe filename from an untrusted source name."""
    base_name = Path(str(original_name)).name
    parsed = Path(base_name)
    safe_stem = sanitize_filename_component(parsed.stem, fallback=fallback)
    safe_suffix = "".join(char for char in parsed.suffix if char.isalnum()).lower()
    suffix = f".{safe_suffix}" if safe_suffix else ""
    return f"{prefix}{safe_stem}{suffix}"


def safe_child_path(base_dir, filename):
    """Resolve a child path and ensure it cannot escape the base directory."""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True, parents=True)
    resolved_base = base_path.resolve()
    candidate = (resolved_base / filename).resolve()

    if os.path.commonpath([str(resolved_base), str(candidate)]) != str(resolved_base):
        raise ValueError("Unsafe output path.")

    return candidate


def _non_negative_int(value, field_name):
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}") from exc

    if normalized < 0:
        raise ValueError(f"{field_name} must be non-negative.")

    return normalized


def _serialize_embedding(embedding):
    np = _require_numpy()
    embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding_array.size == 0:
        raise ValueError("Embedding cannot be empty.")
    if not np.all(np.isfinite(embedding_array)):
        raise ValueError("Embedding contains non-finite values.")
    return embedding_array.tolist()


def _deserialize_embedding(raw_embedding):
    np = _require_numpy()
    embedding_array = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    if embedding_array.size == 0:
        raise ValueError("Embedding cannot be empty.")
    if not np.issubdtype(embedding_array.dtype, np.number):
        raise ValueError("Embedding must contain numeric values.")
    if not np.all(np.isfinite(embedding_array)):
        raise ValueError("Embedding contains non-finite values.")
    return embedding_array


def _to_jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        return _to_jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _config_to_dict(config):
    if config is None:
        return {}
    if isinstance(config, dict):
        return _to_jsonable(config)

    normalized = {}
    for attribute_name in dir(config):
        if attribute_name.startswith("_"):
            continue
        attribute_value = getattr(config, attribute_name)
        if callable(attribute_value):
            continue
        normalized[attribute_name] = _to_jsonable(attribute_value)
    return normalized


def _normalize_face_database(face_database):
    if not isinstance(face_database, dict):
        raise ValueError("face_database must be a dictionary.")

    normalized = {}
    for person_name, record in face_database.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record for {person_name!r}.")

        normalized[str(person_name)] = {
            "embedding": _deserialize_embedding(record.get("embedding")),
            "sample_count": _non_negative_int(record.get("sample_count", 0), "sample_count"),
            "image_count": _non_negative_int(record.get("image_count", 0), "image_count"),
        }
    return normalized


def _json_face_database(face_database):
    normalized = _normalize_face_database(face_database)
    return {
        person_name: {
            "embedding": _serialize_embedding(record["embedding"]),
            "sample_count": record["sample_count"],
            "image_count": record["image_count"],
        }
        for person_name, record in normalized.items()
    }


def _validate_database_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Database payload must be a dictionary.")

    return {
        "version": _non_negative_int(payload.get("version", DATABASE_VERSION), "version"),
        "face_database": _normalize_face_database(payload.get("face_database", {})),
        "processing_stats": _to_jsonable(payload.get("processing_stats", {})),
        "config": _to_jsonable(payload.get("config", {})),
        "timestamp": str(payload.get("timestamp", "")),
    }


class _RestrictedLegacyUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = {
        ("datetime", "datetime"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Disallowed legacy pickle global: {module}.{name}")

        module_obj = __import__(module, fromlist=[name])
        return getattr(module_obj, name)


def _load_legacy_pickle(path):
    with open(path, "rb") as file_obj:
        payload = _RestrictedLegacyUnpickler(file_obj).load()
    return _validate_database_payload(payload)


def save_face_database(face_database, processing_stats=None, config=None, save_path=DEFAULT_DATABASE_PATH):
    """Persist the database in JSON so loading does not execute code."""
    destination = Path(save_path)
    destination.parent.mkdir(exist_ok=True, parents=True)

    payload = {
        "version": DATABASE_VERSION,
        "face_database": _json_face_database(face_database),
        "processing_stats": _to_jsonable(processing_stats or {}),
        "config": _config_to_dict(config),
        "timestamp": datetime.now().isoformat(),
    }

    temp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
    os.replace(temp_path, destination)

    return destination


def load_face_database(database_path=DEFAULT_DATABASE_PATH):
    """
    Load a JSON database or safely migrate a legacy pickle database when needed.
    """
    requested_path = Path(database_path)
    json_path = requested_path

    if requested_path.suffix.lower() in LEGACY_DATABASE_SUFFIXES:
        json_path = requested_path.with_suffix(".json")
    elif requested_path.suffix.lower() != ".json" and not requested_path.exists():
        json_path = requested_path.with_suffix(".json")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        validated = _validate_database_payload(payload)
        validated["database_path"] = str(json_path)
        validated["migrated_from_legacy"] = False
        return validated

    legacy_path = requested_path if requested_path.suffix.lower() in LEGACY_DATABASE_SUFFIXES else json_path.with_suffix(".pkl")
    if not legacy_path.exists():
        raise FileNotFoundError(f"Database file not found: {json_path}")

    validated = _load_legacy_pickle(legacy_path)
    save_face_database(
        validated["face_database"],
        processing_stats=validated["processing_stats"],
        config=validated["config"],
        save_path=json_path,
    )
    validated["database_path"] = str(json_path)
    validated["migrated_from_legacy"] = True
    return validated
