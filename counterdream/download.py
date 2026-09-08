"""Download the published CounterDream weights with pinned size/hash checks."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import requests


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination="artifacts"):
    manifest = json.loads(Path(__file__).with_name("release.json").read_text())
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = (
        "https://github.com/OmuNaman/counterdream/releases/download/" + manifest["tag"]
    )
    for name, expected in manifest["files"].items():
        target = root / name
        if target.parent != root or Path(name).name != name:
            raise ValueError("Invalid release asset name")
        if target.exists():
            if (
                target.stat().st_size == expected["bytes"]
                and sha256(target) == expected["sha256"]
            ):
                print(f"Verified existing {name}")
                continue
            raise FileExistsError(
                f"{target} differs from this release. Choose a new destination."
            )
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=root, suffix=".download", delete=False
            ) as output:
                temporary = Path(output.name)
                size = 0
                digest = hashlib.sha256()
                with requests.get(
                    f"{base}/{name}", stream=True, timeout=(30, 120)
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        size += len(chunk)
                        if size > expected["bytes"]:
                            raise ValueError(f"Unexpected file size: {name}")
                        digest.update(chunk)
                        output.write(chunk)
            if size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Release checksum mismatch: {name}")
            temporary.replace(target)
            print(f"Downloaded and verified {name} ({size:,} bytes)")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Ready. Checkpoint and seeds: {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="artifacts")
    download(parser.parse_args().destination)
