import hashlib
import json

import pytest

from counterdream import download as module


def release(monkeypatch, tmp_path, payload, corrupt=False):
    manifest = tmp_path / "release.json"
    manifest.write_text(
        json.dumps(
            {
                "tag": "v0.1.0",
                "files": {
                    "model.pt": {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(module, "__file__", str(tmp_path / "download.py"))

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"X" * len(payload) if corrupt else payload

    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(module.requests, "get", get)
    return calls


def test_verified_release_is_reused_without_network(monkeypatch, tmp_path):
    payload = b"released model fixture"
    calls = release(monkeypatch, tmp_path, payload)
    target = tmp_path / "downloaded"
    module.download(target)
    module.download(target)
    assert (target / "model.pt").read_bytes() == payload
    assert len(calls) == 1


def test_corrupt_release_never_becomes_a_checkpoint(monkeypatch, tmp_path):
    release(monkeypatch, tmp_path, b"released model fixture", corrupt=True)
    target = tmp_path / "downloaded"
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.download(target)
    assert list(target.iterdir()) == []
