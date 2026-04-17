import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dtw'))
from load_data import load_multi_serve

bones = [
    ("head", "chest"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_hip", "right_hip"),
    ("chest", "left_hip"),
    ("chest", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
]

# Accept a serve name as an optional command-line argument, e.g. "serve5"
serve_name = sys.argv[1] if len(sys.argv) > 1 else "serve3"
csv_path = os.path.join(
    os.path.dirname(__file__),
    "markers", "unmarked_edited", f"{serve_name}.csv"
)
if not os.path.exists(csv_path):
    print(f"File not found: {csv_path}")
    sys.exit(1)

markers = load_multi_serve(csv_path)
frames = markers["frames"]
n_frames = len(frames)
marker_names = [k for k in markers if k != "frames"]


def get_pos(joint, frame_idx):
    m = markers[joint]
    return float(m["TX"][frame_idx]), float(m["TY"][frame_idx]), float(m["TZ"][frame_idx])


# Fixed axis limits based on full data range
all_x = np.concatenate([markers[m]["TX"] for m in marker_names])
all_y = np.concatenate([markers[m]["TY"] for m in marker_names])
all_z = np.concatenate([markers[m]["TZ"] for m in marker_names])

all_x = all_x[~np.isnan(all_x)]
all_y = all_y[~np.isnan(all_y)]
all_z = all_z[~np.isnan(all_z)]


def padded_limits(data, pad=0.08):
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


x_lim = padded_limits(all_x)
y_lim = padded_limits(all_y)
z_lim = padded_limits(all_z)

x_range = x_lim[1] - x_lim[0]
y_range = y_lim[1] - y_lim[0]
z_range = z_lim[1] - z_lim[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def apply_axes():
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


apply_axes()


def update(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f"{serve_name}  —  Frame {int(frames[frame_idx])}")

    for joint in marker_names:
        x, y, z = get_pos(joint, frame_idx)
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(x, y, z, s=20)

    for start, end in bones:
        if start not in markers or end not in markers:
            continue
        x0, y0, z0 = get_pos(start, frame_idx)
        x1, y1, z1 = get_pos(end, frame_idx)
        if any(np.isnan(v) for v in [x0, y0, z0, x1, y1, z1]):
            continue
        ax.plot([x0, x1], [y0, y1], [z0, z1], "b-", linewidth=1.5)


ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.show()
