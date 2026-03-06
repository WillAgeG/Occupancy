import os
import pathlib

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from project2d.scripts import project_dataset


def add_title_and_legend(array, n_tiles, legend, titles):
    """
    Заголовки над тайлами + цветная легенда справа.
    Вход/выход: BGR (OpenCV).
    """

    H, W, _ = array.shape
    tile_size = W // n_tiles

    if titles is not None and len(titles) != n_tiles:
        msg = f"Expected {n_tiles} titles, but got {len(titles)}."
        raise ValueError(msg)

    # --- BGR -> RGB ---
    img_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    # --- Figure (шире под легенду) ---
    fig, ax = plt.subplots(figsize=((W + 156) / 100, (H + 40) / 100))

    # nearest чтобы не размывались цвета
    ax.imshow(img_rgb, interpolation="nearest")
    ax.axis("off")

    # --- Настройка layout ---
    fig.subplots_adjust(
        left=0.0,
        right=0.82,   # место под легенду
        top=0.92,     # место сверху под titles
        bottom=0.02
    )

    # --- Titles над изображением ---
    if titles is not None:
        for i in range(n_tiles):
            cx = i * tile_size + tile_size // 2
            ax.text(cx, -8, titles[i], ha="center", va="bottom", fontsize=10, color="black")

    # --- Цветная легенда ---
    ax.legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)

    # --- Render canvas ---
    fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())
    result = buf[:, :, :3].copy()

    plt.close(fig)

    # --- RGB -> BGR ---
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result


def get_legend(color_lut, mappings):
    label_lut = mappings["labels"]
    learning_map_inv = mappings["learning_map_inv"]
    learning_map = mappings["learning_map"]
    legend_patches = []

    for idx in range(len(set(learning_map.values()))):
        x_idx = learning_map_inv[idx]
        name = label_lut[x_idx]
        if name == "invalid":
            continue
        bgr = color_lut[idx].astype(np.uint8)
        rgb = bgr[::-1] / 255.0

        legend_patches.append(
            mpatches.Patch(color=rgb, label=name)
        )

    return legend_patches


def visualize(arrays: list[np.ndarray], mappings: dict, titles: list[str] | None = None) -> np.array:
    assert isinstance(arrays, list)
    assert len(arrays) > 0

    _, color_lut = project_dataset.build_mappings(mappings)

    num_tiles = len(arrays)
    res = arrays[0]
    res = color_lut[res]
    if num_tiles > 1:
        for i in range(1, num_tiles):
            tmp = color_lut[arrays[i]]
            filler = np.full((res.shape[0], 6, 3), 255, dtype=np.uint8)
            res = np.concatenate([res, filler, tmp], axis=1)

    legend = get_legend(color_lut, mappings)
    res = add_title_and_legend(res, num_tiles, legend, titles)

    return res


def save_visualization(array: np.ndarray, input_path: str, out_dir: str, cml_logger=None, epoch=0) -> None:
    filename, _extension = os.path.splitext(pathlib.Path(input_path).name)
    sequence = pathlib.Path(input_path).parent.parent.name
    out_filename = os.path.join(out_dir, "sequences", sequence, filename + ".png")
    pathlib.Path(pathlib.Path(out_filename).parent).mkdir(parents=True, exist_ok=True)
    if cml_logger is not None:
        cml_logger.report_image(
            title="Validation Results",
            series=os.path.join(sequence, filename + ".png"),
            iteration=epoch,
            image=cv2.cvtColor(array, cv2.COLOR_BGR2RGB),
        )
    cv2.imwrite(out_filename, array)
