"""How do DIGIT images change with the LED illumination?

Mode 1 - recordings (default): compare recording_test.touch (illumination changed in the
         OpenTouch dashboard) against the contactless reference recording.touch.
    python analyze_illumination.py
    python analyze_illumination.py --test datasets/recording_test.touch --ref datasets/recording.touch

Mode 2 - sweep: analyse an intensity sweep captured with collect_digit.py
    python analyze_illumination.py --sweep outputs/collected/illum_sweep_*.h5

PNG outputs go to outputs/illumination/.
"""
import argparse
import os

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from save_interesting import contact_score, diff_heatmap, label, smooth
from touch_loader import load_touch

OUT = "outputs/illumination"
CH = {"B": (0, "#3b6fb6"), "G": (1, "#3f9c5a"), "R": (2, "#d1495b")}   # frames are BGR


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def channel_means(frames):
    return frames.reshape(len(frames), -1, 3).mean(1)


def saturation(frames):
    return (frames.reshape(len(frames), -1) >= 250).mean(1) * 100


def detect_steps(means, fps, thresh=2.0):
    """Illumination steps = jumps in the slowly varying per-channel brightness."""
    sm = np.stack([smooth(means[:, c], 9) for c in range(3)], 1)
    lag = int(fps // 2)
    jump = np.abs(sm[lag:] - sm[:-lag]).max(1)
    idx = np.where(jump > thresh)[0]
    steps, last = [], -10 * lag
    for i in idx:
        if i - last > lag:
            steps.append(i + lag // 2)
        last = i
    return steps


# --------------------------------------------------------------------- recordings
def analyse_recordings(test_path, ref_path):
    test, ref = load_touch(test_path), load_touch(ref_path)
    tname = os.path.basename(test_path)
    print(f"test LED cfg={test.config['sensors'][0]['rgb']}  ref LED cfg={ref.config['sensors'][0]['rgb']}")

    ref_bg = np.median(ref.frames[:60], 0).astype(np.float32)
    test_bg = np.median(test.frames[:30], 0).astype(np.float32)
    m_test, m_ref = channel_means(test.frames), channel_means(ref.frames)
    score = smooth(contact_score(test.frames, test_bg))
    steps = detect_steps(m_test, test.fps)

    # 1. per-channel brightness over time ------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1.3]})
    for k, (c, col) in CH.items():
        axs[0].plot(test.timestamps, m_test[:, c], color=col, lw=1.5, label=f"{k} – {tname}")
        axs[0].axhline(m_ref[:60, c].mean(), color=col, ls="--", lw=1, alpha=0.7)
    for s in steps:
        for a in axs:
            a.axvline(test.timestamps[s], color="#888", ls=":", lw=1)
    axs[0].set(ylabel="mean pixel value", title="Per-channel brightness (dashed = contactless reference recording.touch)")
    axs[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    axs[1].plot(test.timestamps, score, color="#555", lw=1.2)
    axs[1].set(xlabel="time [s]", ylabel="contact score")
    for a in axs:
        style(a)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "brightness_timeline.png"), dpi=130)
    plt.close(fig)

    # 2. image comparison: reference vs test background vs difference --------------
    diff = test_bg - ref_bg
    amp = max(np.abs(diff).max(), 1)
    diff_vis = np.clip(128 + diff / amp * 127, 0, 255).astype(np.uint8)
    contact_i = int(np.argmax(score))
    panel = np.hstack([
        label(ref_bg.astype(np.uint8), "ref (no contact)"),
        label(test_bg.astype(np.uint8), "test (no contact)"),
        label(diff_vis, f"test-ref (+/-{amp:.0f})"),
        label(test.frames[contact_i], f"test contact t={test.timestamps[contact_i]:.1f}s"),
        label(diff_heatmap(test.frames[contact_i], test_bg), "contact |diff|"),
    ])
    cv2.imwrite(os.path.join(OUT, "compare_frames.png"), panel)

    # 3. histograms -----------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    bins = np.arange(257)
    for ax, (k, (c, col)) in zip(axs, CH.items()):
        ax.hist(ref_bg[..., c].ravel(), bins, color="#999", alpha=0.6, label="reference")
        ax.hist(test_bg[..., c].ravel(), bins, color=col, alpha=0.6, label="test")
        ax.set(title=f"{k} channel", xlabel="pixel value", xlim=(0, 255))
        style(ax)
    axs[0].set_ylabel("pixel count")
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "histograms.png"), dpi=130)
    plt.close(fig)

    # 4. per-channel ratio maps: where did the light change? -------------------------
    fig, axs = plt.subplots(1, 3, figsize=(9, 4.2))
    for ax, (k, (c, _)) in zip(axs, CH.items()):
        r = (test_bg[..., c] + 1) / (ref_bg[..., c] + 1)
        im = ax.imshow(r, cmap="RdBu_r", vmin=0.8, vmax=1.2)
        ax.set(title=f"{k}: test / ref", xticks=[], yticks=[])
    fig.colorbar(im, ax=axs, shrink=0.8, label="brightness ratio")
    fig.savefig(os.path.join(OUT, "ratio_maps.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 5. frames around each detected step --------------------------------------------
    if steps:
        strip = [label(test.frames[max(0, s - 15)], f"before {test.timestamps[max(0, s-15)]:.1f}s")
                 for s in steps]
        strip += [label(test.frames[min(len(test) - 1, s + 15)], f"after {test.timestamps[min(len(test)-1, s+15)]:.1f}s")
                  for s in steps]
        cv2.imwrite(os.path.join(OUT, "steps_before_after.png"), np.hstack(strip))

    # numeric summary
    print("\nmean BGR  reference(no contact):", m_ref[:60].mean(0).round(1))
    print("mean BGR  test (first 30 frames): ", m_test[:30].mean(0).round(1))
    print("mean BGR  test (whole file):      ", m_test.mean(0).round(1),
          " range per channel:", (m_test.max(0) - m_test.min(0)).round(1))
    print("std of pixels  ref / test:", ref_bg.std().round(1), "/", test_bg.std().round(1))
    print("saturated pixels (>=250) test max: %.3f%%" % saturation(test.frames).max())
    print("illumination steps detected at t =", [round(test.timestamps[s], 2) for s in steps])
    print("strongest contact at t = %.2fs" % test.timestamps[contact_i])


# --------------------------------------------------------------------- sweep
def analyse_sweep(path):
    with h5py.File(path, "r") as f:
        frames = f["frames"][()]
        levels = f["levels"][()]               # (N,3) rgb intensity per frame
        phase = f["phase"][()].astype(str)    # 'free' or 'contact'
        simulated = bool(f.attrs.get("simulated", False))
    tag = " (SIMULATED)" if simulated else ""
    keys = sorted({tuple(int(v) for v in l) for l in levels}, key=lambda t: (sum(t), t))
    rows, stats = [], []
    for k in keys:
        sel = np.all(levels == k, 1)
        free = frames[sel & (phase == "free")]
        cont = frames[sel & (phase == "contact")]
        bg = np.median(free, 0).astype(np.float32)
        m = bg.reshape(-1, 3).mean(0)
        sat = (bg >= 250).mean() * 100
        noise = free.astype(np.float32).std(0).mean()
        if len(cont):
            c = cont[np.argmax(contact_score(cont, bg))]
            contrast = np.percentile(np.abs(c.astype(np.float32) - bg).mean(2), 99)
        else:
            c, contrast = bg.astype(np.uint8), np.nan
        stats.append((k, *m, sat, noise, contrast))
        rows.append(np.vstack([label(bg.astype(np.uint8), f"rgb={k}"),
                               label(c, "contact"),
                               label(diff_heatmap(c, bg), "|diff|")]))
    cv2.imwrite(os.path.join(OUT, "sweep_grid.png"), np.hstack(rows))

    x = np.arange(len(keys))
    xt = [f"{k[0]},{k[1]},{k[2]}" for k in keys]
    st = np.array([s[1:] for s in stats], float)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for kname, (c, col) in CH.items():
        axs[0].plot(x, st[:, c], "o-", color=col, label=kname)
    axs[0].set(title="mean brightness" + tag, ylabel="pixel value")
    axs[0].legend(frameon=False)
    axs[1].plot(x, st[:, 3], "o-", color="#555", label="saturated %")
    axs[1].plot(x, st[:, 4], "s--", color="#999", label="temporal noise (std)")
    axs[1].set(title="saturation & noise")
    axs[1].legend(frameon=False)
    axs[2].plot(x, st[:, 5], "o-", color="#7b4fa0")
    axs[2].set(title="contact contrast (99th pct |diff|)")
    for a in axs:
        a.set_xticks(x, xt, rotation=45, fontsize=7)
        a.set_xlabel("LED intensity R,G,B")
        style(a)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sweep_curves.png"), dpi=130)
    plt.close(fig)
    for s in stats:
        print(f"rgb={s[0]}  BGR=({s[1]:.0f},{s[2]:.0f},{s[3]:.0f})  sat={s[4]:.2f}%  "
              f"noise={s[5]:.2f}  contrast={s[6]:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="datasets/recording_test.touch")
    ap.add_argument("--ref", default="datasets/recording.touch")
    ap.add_argument("--sweep", help="h5 file from collect_digit.py")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.sweep:
        analyse_sweep(args.sweep)
    else:
        analyse_recordings(args.test, args.ref)
    print("PNGs in", OUT)


if __name__ == "__main__":
    main()
