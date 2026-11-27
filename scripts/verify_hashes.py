"""Check the exact local acceptance build against both public distribution channels."""

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen


def verify(built, github, version):
    for attempt in range(10):
        try:
            with urlopen(
                f"https://pypi.org/pypi/bornlens/{version}/json", timeout=20
            ) as response:
                artifacts = json.load(response)["urls"]
            break
        except HTTPError as error:
            if error.code != 404 or attempt == 9:
                raise
            time.sleep(8)
    files = {p.name: p for p in built.iterdir() if p.is_file()}
    if len(files) != 2 or set(files) != {x["filename"] for x in artifacts}:
        raise ValueError("published file inventory differs from the verified build")
    for item in artifacts:
        name = item["filename"]
        expected = hashlib.sha256(files[name].read_bytes()).hexdigest()
        with urlopen(item["url"], timeout=30) as response:
            online = hashlib.sha256(response.read()).hexdigest()
        if {
            expected,
            online,
            item["digests"]["sha256"],
            hashlib.sha256((github / name).read_bytes()).hexdigest(),
        } != {expected}:
            raise ValueError(f"digest mismatch for {name}")
        print(name, expected)


if __name__ == "__main__":
    verify(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
