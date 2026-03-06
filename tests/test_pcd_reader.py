import json

import numpy as np

from project2d.lib.readers.pcd_reader import PCDReader


def test_read_label_pointwise_list(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(["road", "vehicle"]))

    labels = PCDReader.read_label(label_path, points, return_type="int")

    assert labels.tolist() == [1, 3]


def test_read_label_pointwise_pairs(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
    label_path = tmp_path / "labels.json"
    label_path.write_text(
        json.dumps(
            {
                "point_labels": [
                    {"point_id": 0, "class": "road"},
                    {"point_id": 2, "label": "vehicle"},
                ]
            }
        )
    )

    labels = PCDReader.read_label(label_path, points, return_type="int")

    assert labels.tolist() == [1, 0, 3]
