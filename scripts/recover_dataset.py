"""Recover the complete nested dataset copy from an interrupted Kaggle ZIP.

The Kaggle archive stores duplicate dataset trees. This optional recovery tool
reads local ZIP/ZIP64 headers, checks decompressed sizes and CRC32 values, and
requires the known full image counts before accepting a recovered archive.
Normal downloads should use download_data.sh instead.
"""
from collections import Counter
from pathlib import Path
import struct
import zipfile
import zlib

source = Path("data/chest-xray-pneumonia.zip")
target = Path("data/chest-xray-recovered.zip")
prefix = "chest_xray/chest_xray/"
expected = {"train": 5216, "val": 16, "test": 624}
counts = Counter()
with source.open("rb") as f, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as out:
    while True:
        h = f.read(30)
        if len(h) != 30 or h[:4] != b"PK\x03\x04":
            break
        v = struct.unpack("<4s5H3I2H", h)
        _, _, flags, method, _, _, crc, compressed, size, namelen, extralen = v
        name = f.read(namelen).decode("utf-8")
        extra = f.read(extralen)
        if flags & 8:
            raise ValueError("Streaming data descriptors are not supported")
        pos = 0
        while pos+4 <= len(extra):
            tag, n = struct.unpack_from("<HH", extra, pos)
            chunk = extra[pos+4:pos+4+n]; pos += 4+n
            if tag == 1:
                j = 0
                if size == 0xffffffff:
                    size = struct.unpack_from("<Q", chunk, j)[0]; j += 8
                if compressed == 0xffffffff:
                    compressed = struct.unpack_from("<Q", chunk, j)[0]
        if f.tell()+compressed > source.stat().st_size:
            break
        wanted = name.startswith(prefix) and name.lower().endswith(".jpeg")
        if not wanted:
            f.seek(compressed, 1)
            continue
        raw = f.read(compressed)
        payload = zlib.decompress(raw, -15) if method == 8 else raw if method == 0 else None
        if payload is None or len(payload) != size or zlib.crc32(payload) != crc:
            raise ValueError(f"Corrupt archive member: {name}")
        relative = name[len(prefix):]
        counts[relative.split("/")[0]] += 1
        out.writestr("chest_xray/"+relative, payload)
if dict(counts) != expected:
    target.unlink()
    raise ValueError(f"Incomplete dataset copy: {counts}, expected {expected}")
print(f"CRC-verified complete dataset: {dict(counts)} -> {target}")
