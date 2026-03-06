import os
import argparse
import numpy as np
import rerun as rr

from lib.common import label_lut
from lib.readers import pcd_reader


def visualize_with_rerun(
    home_folder: str,
    scene_folder: str,
    key_frame = "frame_000046",
    app_id: str = "lidar_boxes_rerun",
):
    
    reader = pcd_reader.PCDReader()

    points = reader.read_cloud(os.path.join(home_folder, scene_folder, "lidar", key_frame+".pcd"), xyz=True)
    folder = os.listdir(os.path.join(home_folder, scene_folder, "export-from-admin-panel"))[0]
    point_labels = reader.read_label(os.path.join(home_folder, scene_folder, "export-from-admin-panel", folder, key_frame+".json"), points, default_label="unlabeled")
    
    boxes = reader.get_boxes(os.path.join(home_folder, scene_folder, "export-from-admin-panel", folder, key_frame+".json"))
    labels = np.array([label_lut.label2color_annotations[lab] for lab in point_labels], dtype=np.uint8)

    boxes = reader.get_boxes(os.path.join(home_folder, data_root, "export-from-admin-panel", folder, FRAME + ".json"))
    labels = np.array([label_lut.COLOR_LUT[lab] for lab in point_labels], dtype=np.uint8)

    rr.init(app_id, spawn=True)
    rr.log("world", rr.ViewCoordinates.RDF)

    rr.log(
        "world/points",
        rr.Points3D(points, colors=labels, radii=0.03),
    )

    if boxes:
        centers = np.stack([b.center for b in boxes], axis=0)
        half_sizes = np.stack([b.half_extents for b in boxes], axis=0)
        quats = np.stack([b.R.as_quat() for b in boxes], axis=0)
        ids = np.stack([b.object_id for b in boxes], axis=0)
        box_labels = [b.label for b in boxes]
        box_labels_and_ids = [str(id) + "-" + b.label for b, id in zip(boxes, ids)]
        box_colors = np.stack([label_lut.label2color_annotations[lab] for lab in box_labels], axis=0)

        rr.log(
            "world/boxes",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                rotations=rr.Quaternion(xyzw=quats),
                colors=box_colors,
                labels=box_labels_and_ids,
                fill_mode=2
            ),
        )


if __name__ == "__main__":
    # Example – adapt to your actual paths:
    ap = argparse.ArgumentParser()
    ap.add_argument("--home-folder", type=str, default="/downloaded_data/datasets/occupancy/annotation", help="Path to dataset for visualization (gt annotations)")
    ap.add_argument("--scene-folder", type=str, default="kc2-116__logs__2025-07-16__07-30-22__situation__labeling__dev_test__43", help="subfolder with a scene")
    args = ap.parse_args()

    visualize_with_rerun(args.home_folder, args.scene_folder)
