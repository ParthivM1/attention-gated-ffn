from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_FILES = [
    "modal_plain_teacher_10k2k_e20_best.pth",
    "old_modal_geovit_teacher_init_10k2k_e20_best.pth",
    "old_modal_geovit_anchorrefine_warm075_head_anchorfeat_kd_10k2k_e10_best.pth",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage strong local GeoViT checkpoints into checkpoints/bootstrap for Modal.")
    parser.add_argument("--source-dir", default="tmp/modal_geovit_results")
    parser.add_argument("--dest-dir", default="checkpoints/bootstrap")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str | int]] = []
    missing: list[str] = []
    for name in args.files:
        src = source_dir / name
        dst = dest_dir / name
        if not src.exists():
            missing.append(name)
            continue
        shutil.copy2(src, dst)
        copied.append({"name": name, "bytes": dst.stat().st_size, "dest": str(dst.as_posix())})

    manifest = {
        "source_dir": str(source_dir.as_posix()),
        "dest_dir": str(dest_dir.as_posix()),
        "copied": copied,
        "missing": missing,
    }
    (dest_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
