"""Loader for .touch files recorded with the OpenTouch dashboard (DIGIT sensor).

File layout (HDF5):
    config            JSON string (sensor name, serial, rgb LED setting, resolution)
    chunk_start_times float64[n_chunks]
    chunk_end_times   float64[n_chunks]
    sensor_chunks     vlen uint8[n_chunks]  -> custom binary blob per chunk

Binary blob of each chunk (little endian):
    u32 n_sensors, u32 len, sensor_name
    u32 n_streams, u32 len, stream_name
    u32 n_frames
    repeated n_frames times:
        u32 record_len
        char[32] stream_name (null padded)
        f64 timestamp (s since recording start)
        u32 height(rows), u32 width(cols), u32 channels
        u8[height*width*channels] pixels (BGR, as delivered by OpenCV)
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class TouchRecording:
    path: str
    config: dict
    frames: np.ndarray      # (N, H, W, 3) uint8, BGR
    timestamps: np.ndarray  # (N,) float64 seconds

    @property
    def fps(self) -> float:
        if len(self.timestamps) < 2:
            return 30.0
        return float(1.0 / np.median(np.diff(self.timestamps)))

    def __len__(self) -> int:
        return len(self.frames)


def _read_str(b: bytes, o: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", b, o)
    o += 4
    return b[o:o + n].decode(errors="replace"), o + n


def _parse_chunk(b: bytes):
    o = 4                                   # n_sensors (always 1 here)
    _sensor, o = _read_str(b, o)
    o += 4                                  # n_streams
    _stream, o = _read_str(b, o)
    (n,) = struct.unpack_from("<I", b, o)
    o += 4
    frames, ts = [], []
    for _ in range(n):
        (rec_len,) = struct.unpack_from("<I", b, o)
        o += 4
        body = o
        (t,) = struct.unpack_from("<d", b, body + 32)
        h, w, c = struct.unpack_from("<III", b, body + 40)
        px = np.frombuffer(b, np.uint8, h * w * c, body + 52).reshape(h, w, c)
        frames.append(px)
        ts.append(t)
        o = body + rec_len
    return frames, ts


def load_touch(path: str) -> TouchRecording:
    with h5py.File(path, "r") as f:
        config = json.loads(f["config"][()])
        frames, ts = [], []
        for chunk in f["sensor_chunks"][()]:
            fr, t = _parse_chunk(chunk.tobytes())
            frames += fr
            ts += t
    return TouchRecording(path, config, np.stack(frames), np.asarray(ts))


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        r = load_touch(p)
        print(f"{p}: {len(r)} frames, {r.frames.shape[1:]}, "
              f"{r.timestamps[-1]:.1f}s @ {r.fps:.1f} fps, config={r.config['sensors'][0]}")
