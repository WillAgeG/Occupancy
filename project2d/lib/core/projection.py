import numpy as np


def project_to_bev(voxel_labels: np.ndarray, car_height_m: float = 3.0, voxel_size_z: int = 0.2,
                   unlabeled_class: int = 0) -> np.array:
    """Проецируем 3D voxel grid в 2D BEV карту (RGB).

    Args:
        voxel_labels (np.ndarray): np.array [W, D, H] с лейблами
        car_height_m (float, optional): Высота ego_vehicle в метрах. Defaults to 3.0.
        voxel_size_z (int, optional): Высота вокселей в метрах. Defaults to 0.2.
        unlabeled_class (int, optional): Индекс нулевого/пустого/неразмеченного класса. Defaults to 0.

    Returns:
        np.array: bev_classes [W, D] с индексами классов в пикселях/элементах массива
    """
    assert voxel_labels.ndim == 3, "Ожидается тензор [W, D, H]"
    W, D, H = voxel_labels.shape

    car_height_vox = min(int(car_height_m / voxel_size_z), H)
    assert car_height_vox > 0, "Высота машины в вокселях должна быть больше нуля"

    # Берём только нижние car_height_vox слоёв
    v = voxel_labels[:, :, :car_height_vox]  # [W, D, H']

    # Маска валидных вокселей
    mask_valid = (v != unlabeled_class)  # bool [W, D, H']

    # Проверяем, есть ли вообще валидные
    has_valid = mask_valid.any(axis=2)  # [W, D]

    # индекс верхнего валидного по z (если нет валидного - argmax даст 0, но мы учтём это ниже)
    last_idx = car_height_vox - 1 - np.argmax(mask_valid[:, :, ::-1], axis=2)  # [W, D]
    chosen = np.take_along_axis(v, last_idx[..., None], axis=2)[..., 0]  # [W, D]

    # если нет валидного вокселя → ставим unlabeled_class (пусто)
    bev_classes = np.where(has_valid, chosen, unlabeled_class).astype(np.int32)

    # Это важно для соответствия проекции координатам в вокселях. Как минимум в semantic kitti так
    bev_classes = bev_classes.T

    return bev_classes


def check_max_class(bev_classes: np.array, max_class_id: int, unlabeled_class: int = 0) -> np.array:
    if bev_classes.max() > max_class_id:
        # всё что больше доступного class id -- обнулим (чёрный)
        bev_classes = np.where(bev_classes <= max_class_id, bev_classes, unlabeled_class)

    return bev_classes
