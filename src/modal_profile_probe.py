from __future__ import annotations

import argparse
import os
import subprocess
import time
import tomllib
from pathlib import Path


def load_profiles(config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    return [key for key, value in data.items() if isinstance(value, dict) and "token_id" in value]


def probe_profile(profile: str, timeout: float) -> dict[str, object]:
    env = os.environ.copy()
    env["MODAL_PROFILE"] = profile
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            ["modal", "app", "list"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start
        output = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        return {
            "profile": profile,
            "status": "ok" if proc.returncode == 0 else "error",
            "returncode": proc.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "preview": output[:6],
        }
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return {
            "profile": profile,
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 3),
            "preview": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Modal profiles with a cheap network call.")
    parser.add_argument("--config", type=Path, default=Path.home() / ".modal.toml")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("profiles", nargs="*", help="Specific profile names to probe")
    args = parser.parse_args()

    profiles = args.profiles or load_profiles(args.config)
    if not profiles:
        raise SystemExit("No Modal profiles found to probe.")

    for profile in profiles:
        print(probe_profile(profile, args.timeout), flush=True)


if __name__ == "__main__":
    main()
