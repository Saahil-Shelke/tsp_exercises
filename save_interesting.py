"""Automatically find and save the most interesting (strongest-contact) frames.

The contactless recording.touch provides the reference gel image.
For every object recording each frame is scored by how much it differs from the
no-contact background; peaks of that score are contact events.

    python save_interesting.py                 # all object files in datasets/
    python save_interesting.py --top 5

Outputs (outputs/frames/):
    <obj>_fXXXX.png            raw frame (as recorded)
    <obj>_fXXXX_panel.png      reference | contact | difference heat-map
    <obj>_score.png            contact score over time with the picked frames
    overview.png               best frame of every object side by side
"""
import argparse
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from touch_loader import load_touch

OUT = "outputs/frames"


def background(frames, n=30):
    """Per-file no-contact background = median of the first n frames."""
    return np.median(frames[:n], axis=0).astype(np.float32)


def contact_score(frames, bg):
    """Mean absolute difference to the background, per frame (robust to small noise)."""
    d = np.abs(frames.astype(np.float32) - bg).mean(axis=3)   # (N,H,W)
    d = d.reshape(len(frames), -1)
    return np.percentile(d, 99, axis=1)                          # the strongest 1% of pixels


def smooth(score, w=7):
    """Running median: removes single-frame glitches (corrupted/dropped frames)."""
    pad = np.pad(score, w // 2, mode="edge")
    return np.array([np.median(pad[i:i + w]) for i in range(len(score))])


def pick_peaks(score, k, min_gap):
    order = np.argsort(score)[::-1]
    chosen = []
    for i in order:
        if all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return sorted(chosen)


def diff_heatmap(frame, bg):
    d = np.abs(frame.astype(np.float32) - bg).mean(axis=2)
    d = np.clip(d / max(d.max(), 1) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    ref = load_touch(os.path.join(args.data, "recording.touch"))
    ref_bg = background(ref.frames, 60)
    cv2.imwrite(os.path.join(OUT, "reference_no_contact.png"), ref_bg.astype(np.uint8))

    best = []
    for path in sorted(glob.glob(os.path.join(args.data, "recording_*.touch"))):
        obj = os.path.basename(path)[len("recording_"):-len(".touch")]
        rec = load_touch(path)
        bg = background(rec.frames)
        raw = contact_score(rec.frames, bg)
        score = smooth(raw)
        glitches = np.where(raw - score > 3 * np.std(raw - score) + 10)[0]
        picks = pick_peaks(score, args.top, min_gap=int(rec.fps))   # >= 1 s apart

        for i in picks:
            f = rec.frames[i]
            cv2.imwrite(os.path.join(OUT, f"{obj}_f{i:04d}.png"), f)
            panel = np.hstack([label(bg.astype(np.uint8), "no contact"),
                               label(f, f"{obj} t={rec.timestamps[i]:.2f}s"),
                               label(diff_heatmap(f, bg), "|diff|")])
            cv2.imwrite(os.path.join(OUT, f"{obj}_f{i:04d}_panel.png"), panel)

        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot(rec.timestamps, raw, lw=0.8, color="#bbbbbb", label="raw")
        ax.plot(rec.timestamps, score, lw=1.4, color="#3b6fb6", label="median-filtered")
        ax.scatter(rec.timestamps[picks], score[picks], color="#d1495b", zorder=3, label="saved")
        ax.set(xlabel="time [s]", ylabel="contact score", title=f"{obj}: deviation from no-contact")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{obj}_score.png"), dpi=120)
        plt.close(fig)

        top = picks[int(np.argmax(score[picks]))]
        best.append(label(rec.frames[top], obj))
        print(f"{obj:6s} frames saved: {picks}  (best t={rec.timestamps[top]:.2f}s)  "
              f"glitch frames ignored: {glitches.tolist()}")

    cv2.imwrite(os.path.join(OUT, "overview.png"),
                np.hstack([label(ref_bg.astype(np.uint8), "no contact")] + best))
    print("outputs in", OUT)


if __name__ == "__main__":
    main()
