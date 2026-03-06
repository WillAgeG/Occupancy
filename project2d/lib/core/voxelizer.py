import numpy as np


def voxelize(
    points: np.ndarray,
    labels: np.ndarray,
    voxel_size=(0.2, 0.2, 0.2),
    coors_range=(-51.2, -51.2, -1.6, 51.2, 51.2, 4.8),
    reduction="mode",    # 'mode' или 'max'
    ignore_label=0,      # всё с этим label игнорируется
) -> np.ndarray:
    """
    Преобразует точки и метки в плотную воксельную сетку (W, D, H),
    игнорируя метки пустоты (ignore_label).

    Args:
        points: (N, 3) xyz координаты
        labels: (N,) int32 — семантические классы
        voxel_size: (vx, vy, vz)
        coors_range: (xmin, ymin, zmin, xmax, ymax, zmax)
        reduction: 'mode' — мода меток в вокселе,
                   'max' — максимум
        ignore_label: int — метка пустоты, которую нужно игнорировать

    Returns:
        dense_labels: np.ndarray формы [W, D, H] с int32 метками
    """
    points = np.asarray(points)
    labels = np.asarray(labels)

    # Убираем пустоту
    valid_mask = labels != ignore_label
    points = points[valid_mask]
    labels = labels[valid_mask]
    if len(points) == 0:
        raise ValueError("Нет непустых точек после фильтрации меток пустоты.")

    # Координаты вокселей
    x_min, y_min, z_min, x_max, y_max, z_max = coors_range
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_shape = np.round(
        np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / voxel_size
    ).astype(int)
    W, D, H = grid_shape

    coords = np.floor((points[:, :3] - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)

    # Фильтруем точки вне диапазона
    valid_mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < W) &
        (coords[:, 1] >= 0) & (coords[:, 1] < D) &
        (coords[:, 2] >= 0) & (coords[:, 2] < H)
    )
    coords = coords[valid_mask]
    labels = labels[valid_mask]

    flat_ids = np.ravel_multi_index((coords[:, 0], coords[:, 1], coords[:, 2]), grid_shape)
    dense_labels = np.zeros(grid_shape, dtype=np.int32)

    if reduction == "max":
        np.maximum.at(dense_labels.ravel(), flat_ids, labels.astype(np.int32))
        return dense_labels

    # Определяем лейбл по моде
    uniq_ids, inv = np.unique(flat_ids, return_inverse=True)
    # uniq_ids - воксели, inv какому вокселю принадлежит точка

    # Приводим метки к дискретным индексам
    uniq_labels, label_inv = np.unique(labels, return_inverse=True)
    # uniq_labels - лейблы, label_inv - какой лейбл у точки

    # Строим сводную таблицу вокселей и содержащихся в них лейблов
    counts = np.zeros((uniq_ids.size, uniq_labels.size), dtype=np.int32)
    # берём воксель, в котором лежит точка, берём её лейбл, добавляем 1 в counts. Т.е. в таком-то вокселе есть точка
    # с таким лейблом
    np.add.at(counts, (inv, label_inv), 1)

    # Выбираем наиболее частую метку для каждого вокселя
    mode_label_indices = np.argmax(counts, axis=1)
    # это делается, так как лейблы могут нумероваться не с 0
    mode_labels = uniq_labels[mode_label_indices]

    # Формируем финальную dense-карту
    # prod - произведение измерениц массива W*D*H
    dense_labels_flat = np.zeros(grid_shape.prod(), dtype=uniq_labels.dtype)
    # Присваем вокселю самый частый лейбл
    dense_labels_flat[uniq_ids] = mode_labels
    dense_labels = dense_labels_flat.reshape(grid_shape)

    return dense_labels


def voxelize_points(
    points: np.ndarray,
    voxel_size=(0.2, 0.2, 0.2),
    coors_range=(-51.2, -51.2, -1.6, 51.2, 51.2, 4.8),
) -> np.ndarray:
    """
    Вокселизация точек без лейблов.
    Возвращает occupancy grid — 1, если в вокселе есть точки, 0 — если пусто.

    Args:
        points: (N, 3) xyz координаты
        voxel_size: (vx, vy, vz)
        coors_range: (xmin, ymin, zmin, xmax, ymax, zmax)

    Returns:
        occupancy: np.ndarray формы [W, D, H]
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points должны иметь форму (N, 3)")

    # Расчёт формы сетки
    x_min, y_min, z_min, x_max, y_max, z_max = coors_range
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_shape = np.round(
        np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / voxel_size
    ).astype(int)
    W, D, H = grid_shape

    # Координаты вокселей
    coords = np.floor((points[:, :3] - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)

    # Фильтруем точки вне диапазона
    valid_mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < W) &
        (coords[:, 1] >= 0) & (coords[:, 1] < D) &
        (coords[:, 2] >= 0) & (coords[:, 2] < H)
    )
    coords = coords[valid_mask]

    # Заполняем occupancy grid
    occupancy = np.zeros(grid_shape, dtype=np.float32)

    flat_ids = np.ravel_multi_index((coords[:, 0], coords[:, 1], coords[:, 2]), grid_shape)
    occupancy.ravel()[np.unique(flat_ids)] = 1

    return occupancy


def voxelize_points_with_time(
    points: np.ndarray,
    voxel_size=(0.2, 0.2, 0.2),
    coors_range=(-51.2, -51.2, -1.6, 51.2, 51.2, 4.8),
    time_method: str = "mean",  # "mean", "max", "last"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Вокселизация точек с учётом времени.

    Args:
        points: (N, 4) — (x, y, z, t)
        voxel_size: (vx, vy, vz)
        coors_range: (xmin, ymin, zmin, xmax, ymax, zmax)
        time_method: как агрегировать время в вокселе: "mean", "max" или "last"

    Returns:
        occupancy: [W, D, H] — 0 или 1
        time_grid: [W, D, H] — агрегированное время
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points должны иметь форму (N, 4): x, y, z, t")

    x_min, y_min, z_min, x_max, y_max, z_max = coors_range
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_shape = np.round(
        np.array([x_max - x_min, y_max - y_min, z_max - z_min]) / voxel_size
    ).astype(int)
    W, D, H = grid_shape

    coords = np.floor((points[:, :3] - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)
    valid_mask = (
        (coords[:, 0] >= 0) & (coords[:, 0] < W) &
        (coords[:, 1] >= 0) & (coords[:, 1] < D) &
        (coords[:, 2] >= 0) & (coords[:, 2] < H)
    )
    coords = coords[valid_mask]
    times = points[valid_mask, 3]

    occupancy = np.zeros(grid_shape, dtype=np.float32)
    time_grid = np.zeros(grid_shape, dtype=np.float32)

    # Уникальные индексы вокселей
    flat_ids, inverse, counts = np.unique(
        np.ravel_multi_index((coords[:, 0], coords[:, 1], coords[:, 2]), grid_shape),
        return_inverse=True,
        return_counts=True
    )

    occupancy.ravel()[flat_ids] = 1

    # Агрегируем время по вокселю
    if time_method == "mean":
        sums = np.bincount(inverse, weights=times)
        means = sums / counts
        time_grid.ravel()[flat_ids] = means
    elif time_method == "max":
        max_times = np.zeros_like(flat_ids, dtype=np.float32)
        np.maximum.at(max_times, inverse, times)
        time_grid.ravel()[flat_ids] = max_times
    elif time_method == "last":
        # просто последнее встреченное значение
        time_grid.ravel()[flat_ids] = times[np.arange(len(times))]
    else:
        raise ValueError(f"Неизвестный метод {time_method}")

    return occupancy, time_grid
