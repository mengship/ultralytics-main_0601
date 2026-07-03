#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prefix dataset filenames with their type directory name.

Example:
    lower left/330.jpg  -> lower left/lower_left_330.jpg
    lower left/330.json -> lower left/lower_left_330.json

When a JSON file has an imagePath matching a renamed image in the same
directory, imagePath is updated to the new image basename.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


TYPE_DIRS = {
    "left": "left",
    "lower left": "lower_left",
    "lower right": "lower_right",
    "top right": "top_right",
}

DATA_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".json"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefix type directory names onto image/json filenames.")
    parser.add_argument("--root", required=True, help="Root directory containing left/lower left/lower right/top right.")
    parser.add_argument("--apply", action="store_true", help="Actually rename files and update JSON imagePath.")
    return parser.parse_args()


def prefixed_name(prefix: str, name: str) -> str:
    return name if name.startswith(f"{prefix}_") else f"{prefix}_{name}"


def collect_renames(root: Path) -> Tuple[List[Tuple[Path, Path]], Dict[Path, Dict[str, str]]]:
    renames: List[Tuple[Path, Path]] = []
    image_name_maps: Dict[Path, Dict[str, str]] = {}

    for dirname, prefix in TYPE_DIRS.items():
        directory = root / dirname
        if not directory.exists():
            print(f"[WARN] missing directory: {directory}")
            continue

        image_name_maps[directory] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in DATA_EXTS:
                continue

            new_path = path.with_name(prefixed_name(prefix, path.name))
            if new_path == path:
                continue
            if new_path.exists():
                raise FileExistsError(f"target already exists: {new_path}")

            renames.append((path, new_path))
            if path.suffix.lower() in IMAGE_EXTS:
                image_name_maps[directory][path.name] = new_path.name

    return renames, image_name_maps


def update_json_image_path(json_path: Path, image_name_map: Dict[str, str], apply: bool) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read JSON {json_path}: {exc}")
        return False

    old_image_path = data.get("imagePath")
    if not old_image_path:
        return False

    old_basename = Path(str(old_image_path)).name
    new_basename = image_name_map.get(old_basename)
    if not new_basename:
        return False

    data["imagePath"] = new_basename
    print(f"[JSON] {json_path.name}: imagePath {old_basename} -> {new_basename}")
    if apply:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    renames, image_name_maps = collect_renames(root)

    print(f"Root: {root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Files to rename: {len(renames)}")

    for old_path, new_path in renames:
        print(f"[RENAME] {old_path.relative_to(root)} -> {new_path.relative_to(root)}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to modify files.")
        return

    for old_path, new_path in renames:
        old_path.rename(new_path)

    updated_json_count = 0
    for directory, image_name_map in image_name_maps.items():
        for json_path in sorted(directory.glob("*.json")):
            if update_json_image_path(json_path, image_name_map, apply=True):
                updated_json_count += 1

    print(f"\nDone. Renamed files: {len(renames)}")
    print(f"Updated JSON imagePath fields: {updated_json_count}")


if __name__ == "__main__":
    main()
