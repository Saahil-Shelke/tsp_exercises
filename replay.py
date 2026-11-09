"""Replay the frames of a .touch file.

    python replay.py datasets/recording_key.touch            # OpenCV window
    python replay.py datasets/recording_key.touch --speed 0.5
    python replay.py datasets/recording_key.touch --export   # write outputs/replay/<name>.mp4 (no GUI needed)

Keys in the window:  space = pause/resume,  a / d = step back / forward (when paused),
                     s = save current frame as PNG,  q / Esc = quit
"""
import argparse
import os

import cv2
import numpy as np

from touch_loader import load_touch

SCALE = 2  # upscale the 240x320 image so it is easier to see


def overlay(frame, i, n, t):
    img = cv2.resize(frame, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, f"{i+1}/{n}  t={t:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--export", action="store_true", help="write an mp4 instead of opening a window")
    ap.add_argument("--out", default="outputs/replay")
    args = ap.parse_args()
    rec = load_touch(args.file)
    name = os.path.splitext(os.path.basename(args.file))[0]
    n = len(rec)
    print(f"{name}: {n} frames @ {rec.fps:.1f} fps, LED rgb={rec.config['sensors'][0]['rgb']}")
    os.makedirs(args.out, exist_ok=True)

    if args.export:
        h, w = rec.frames.shape[1] * SCALE, rec.frames.shape[2] * SCALE
        path = os.path.join(args.out, f"{name}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), rec.fps * args.speed, (w, h))
        for i in range(n):
            vw.write(overlay(rec.frames[i], i, n, rec.timestamps[i]))
        vw.release()
        print("wrote", path)
        return

    delay = max(1, int(1000 / (rec.fps * args.speed)))
    i, paused = 0, False
    while True:
        cv2.imshow(name, overlay(rec.frames[i], i, n, rec.timestamps[i]))
        k = cv2.waitKey(0 if paused else delay) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            paused = not paused
        elif k == ord("s"):
            p = os.path.join(args.out, f"{name}_f{i:04d}.png")
            cv2.imwrite(p, rec.frames[i])
            print("saved", p)
        elif k == ord("a"):
            i = max(0, i - 1)
            continue
        elif k == ord("d"):
            i = min(n - 1, i + 1)
            continue
        if not paused:
            i = (i + 1) % n
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
