from __future__ import annotations

import numpy as np
import open3d as o3d

from .common import box, label_lut


def visualize_as_voxelgrid(dense_labels, voxel_size=0.2, label_to_color=None):
    xs, ys, zs = np.nonzero(dense_labels > 0)
    labels = dense_labels[xs, ys, zs]
    coords = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    coords = coords * voxel_size

    if label_to_color is None:
        label_to_color = {}
        for lbl in np.unique(labels):
            label_to_color[lbl] = deterministic_color(lbl)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(
        np.array([label_to_color[int(l)] for l in labels])
    )

    # Конвертируем в воксельную сетку
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=voxel_size
    )
    o3d.visualization.draw_geometries([voxel_grid], window_name="Semantic Voxel Grid")


def get_color_lut():
    max_id = max(label_lut.LABEL2INT_extended.values())
    color_lut = np.zeros((max_id + 1, 3), dtype=np.uint8)

    for label, idx in label_lut.LABEL2INT_extended.items():
        rgb = np.array(deterministic_color(label)) * 255
        color_lut[idx] = rgb.astype(np.uint8)

    # для "пустоты"
    color_lut[0] = np.array([0, 0, 0], dtype=np.uint8)
    return color_lut


def deterministic_color(label: str) -> tuple[float, float, float]:
    """
    Stable, perceptually distinct-ish RGB from label using HSV hashing.
    """
    if label == "unlabeled":
        return (0.6, 0.6, 0.6)
    h = (hash(label) % 360) / 360.0
    s = 0.65
    v = 0.95
    # hsv -> rgb
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return (r, g, b)


def colors_for_labels(point_labels: np.ndarray):
    uniq = list(dict.fromkeys(point_labels.tolist()))  # keep order
    lut = {lbl: deterministic_color(lbl) for lbl in uniq}
    colors = np.array([lut[lbl] for lbl in point_labels], dtype=np.float32)
    return colors


def visualize_open3d(
    points: np.ndarray,
    point_labels: np.ndarray,
    obbs: list[box.OBB] | None = None,
    voxel: float | None = None,
) -> None:
    import open3d as o3d

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors_for_labels(point_labels))

    # Optional downsample (only display; labels used only for color anyway)
    geoms = []
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel))
    geoms.append(pcd)

    # Draw OBB wires (colored by label color)
    # Use one lineset per label to avoid too many drawables
    label_to_color = {}
    for lbl in np.unique(point_labels):
        label_to_color[lbl] = deterministic_color(lbl)

    if obbs is not None:
        for b in obbs:
            geoms.append(
                box.obb_to_o3d(b, label_to_color.get(b.label, (0.2, 0.2, 0.2)))
            )

    # Add world axes
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    print("Legend (label -> RGB 0-255):")
    for lbl, c in label_to_color.items():
        print(f"  {lbl} -> ({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})")

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Point Labels (Open3D)",
        width=1280,
        height=800,
        # point_show_normal=False
    )


def visualize_matplotlib(
    points: np.ndarray, point_labels: np.ndarray, max_points: int = 200_000
) -> None:
    """
    Fallback simple 3D scatter. Subsamples for speed if needed.
    """
    import matplotlib.pyplot as plt

    N = points.shape[0]
    if max_points < N:
        idx = np.random.RandomState(42).choice(N, size=max_points, replace=False)
        pts = points[idx]
        lbls = point_labels[idx]
    else:
        pts = points
        lbls = point_labels

    colors = colors_for_labels(lbls)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=colors)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Labels (Matplotlib fallback)")
    plt.show()
