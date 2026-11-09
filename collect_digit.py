"""Collect DIGIT data while sweeping the LED illumination.

For every intensity setting: wait for the LEDs/auto-exposure to settle, record frames
with nothing touching the gel ("free"), then prompt you to press an object and record
again ("contact").

    python collect_digit.py --serial D20758                        # global sweep 0..15
    python collect_digit.py --serial D20758 --mode rgb             # one LED at a time
    python collect_digit.py --serial D20758 --no-contact           # skip the press phase
    python collect_digit.py --simulate                             # no sensor: fake sweep from recording_test.touch

Output: outputs/collected/illum_sweep_<timestamp>.h5 (frames, levels, phase, timestamps)
then:   python analyze_illumination.py --sweep <that file>
"""
import argparse
import os
import time

import h5py
import numpy as np

OUT = "outputs/collected"


def sweep_levels(mode):
    if mode == "global":
        return [(v, v, v) for v in (0, 2, 4, 6, 8, 10, 12, 15)]
    if mode == "rgb":   # each LED alone, then all at full
        return [(15, 0, 0), (0, 15, 0), (0, 0, 15), (8, 8, 8), (15, 15, 15)]
    raise ValueError(mode)


class SimulatedDigit:
    """Stands in for the sensor: scales recorded frames per LED channel. Only for testing the pipeline."""

    def __init__(self, path="datasets/recording_test.touch"):
        from touch_loader import load_touch
        rec = load_touch(path)
        self.free = rec.frames[:30]
        self.contact = rec.frames[230:250]
        self.rgb, self.i, self.pressing = (15, 15, 15), 0, False

    def set_intensity_rgb(self, r, g, b):
        self.rgb = (r, g, b)

    def get_frame(self):
        src = self.contact if self.pressing else self.free
        f = src[self.i % len(src)].astype(np.float32)
        self.i += 1
        gain = np.array([self.rgb[2], self.rgb[1], self.rgb[0]], np.float32) / 15  # BGR
        dark = 12.0                                                                  # sensor black level
        f = dark + (f - dark) * (0.15 + 1.6 * gain)                                  # 15 -> ~1.75x (forces clipping)
        f += np.random.normal(0, 2.0 + 4.0 * (1 - gain.mean()), f.shape)            # darker -> noisier
        return np.clip(f, 0, 255).astype(np.uint8)

    def disconnect(self):
        pass


def open_digit(serial, fps):
    from digit_interface import Digit
    d = Digit(serial, "Finger")
    d.connect()
    d.set_resolution(Digit.STREAMS["QVGA"])
    d.set_fps(Digit.STREAMS["QVGA"]["fps"][f"{fps}fps"])
    print(d.info())
    return d


def grab(d, n):
    frames, ts = [], []
    for _ in range(n):
        frames.append(d.get_frame())
        ts.append(time.time())
    return frames, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="D20758")
    ap.add_argument("--fps", type=int, default=30, choices=(30, 60))
    ap.add_argument("--mode", default="global", choices=("global", "rgb"))
    ap.add_argument("--frames", type=int, default=30, help="frames per phase per level")
    ap.add_argument("--settle", type=int, default=20, help="frames discarded after changing LEDs")
    ap.add_argument("--no-contact", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()

    d = SimulatedDigit() if args.simulate else open_digit(args.serial, args.fps)
    frames, levels, phase, stamps = [], [], [], []
    try:
        for lvl in sweep_levels(args.mode):
            d.set_intensity_rgb(*lvl)
            grab(d, args.settle)
            f, t = grab(d, args.frames)
            frames += f; stamps += t; levels += [lvl] * len(f); phase += ["free"] * len(f)
            print(f"rgb={lvl}: {len(f)} free frames, mean={np.mean(f):.1f}")
            if not args.no_contact:
                if args.simulate:
                    d.pressing = True
                else:
                    input("  press the object onto the gel and hit Enter ...")
                    grab(d, 5)
                f, t = grab(d, args.frames)
                frames += f; stamps += t; levels += [lvl] * len(f); phase += ["contact"] * len(f)
                if args.simulate:
                    d.pressing = False
                else:
                    input("  release the object and hit Enter ...")
    finally:
        d.set_intensity_rgb(15, 15, 15)
        d.disconnect()

    os.makedirs(OUT, exist_ok=True)
    tag = "sim_" if args.simulate else ""
    path = os.path.join(OUT, f"illum_sweep_{tag}{time.strftime('%Y%m%d_%H%M%S')}.h5")
    with h5py.File(path, "w") as h:
        h.create_dataset("frames", data=np.stack(frames), compression="gzip")
        h.create_dataset("levels", data=np.array(levels, np.uint8))
        h.create_dataset("phase", data=np.array(phase, dtype="S7"))
        h.create_dataset("timestamps", data=np.array(stamps) - stamps[0])
        h.attrs.update(serial=args.serial, mode=args.mode, simulated=args.simulate)
    print("saved", path, "\nnext: python analyze_illumination.py --sweep", path)


if __name__ == "__main__":
    main()
