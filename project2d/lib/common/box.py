import dataclasses
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import RigidTransform


@dataclasses.dataclass
class OBB:
    center: np.ndarray       # (3,)
    half_extents: np.ndarray  # (3,)
    R: Rotation            # (3,3) box->world
    object_id: int
    label: str
    volume: float
    attribute: str = None

    def __post_init__(self):
        if not isinstance(self.center, np.ndarray):
            raise TypeError(f"center must be a numpy array, got {type(self.center).__name__}")
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")

        if not isinstance(self.half_extents, np.ndarray):
            raise TypeError(f"half_extents must be a numpy array, got {type(self.half_extents).__name__}")
        if self.half_extents.shape != (3,):
            raise ValueError(f"half_extents must have shape (3,), got {self.half_extents.shape}")

        if not isinstance(self.R, Rotation):
            raise TypeError(f"R must be a numpy array, got {type(self.R).__name__}")

    @staticmethod
    def from_json_entry(entry: dict) -> "OBB":
        psr = entry["psr"]
        sx, sy, sz = psr["scale"]["x"], psr["scale"]["y"], psr["scale"]["z"]
        cx, cy, cz = psr["position"]["x"], psr["position"]["y"], psr["position"]["z"]
        rx, ry, rz = psr["rotation"]["x"], psr["rotation"]["y"], psr["rotation"]["z"]
        R = Rotation.from_euler("zyx", (rz, ry, rx))
        half_extents = 0.5 * np.array([sx, sy, sz], dtype=float)
        center = np.array([cx, cy, cz], dtype=float)
        attr = entry["obj_attr"] if "obj_attr" in entry else None
        return OBB(center, half_extents, R, int(entry["obj_id"]), entry["obj_type"], float(sx*sy*sz), attr)

    def to_json(self,) -> dict:
        entry = {"psr": {},
                "obj_id": None,
                "obj_attr": None,
                "obj_type": None}
        
        entry["psr"]["scale"] = {
            "x": self.half_extents[0] * 2,
            "y": self.half_extents[1] * 2,
            "z": self.half_extents[2] * 2,
        }
        
        entry["psr"]["position"] = {
            "x": self.center[0],
            "y": self.center[1],
            "z": self.center[2],
        }
        
        entry["psr"]["rotation"] = {
            "x": self.R.as_euler("zyx")[2],
            "y": self.R.as_euler("zyx")[1],
            "z": self.R.as_euler("zyx")[0],
        }

        entry["obj_id"] = str(self.object_id)
        entry["obj_attr"] = self.attribute
        entry["obj_type"] = self.label

        return entry
        
    def transformed(self, T: RigidTransform) -> "OBB":
        """
        Apply a world-space rigid transform to this OBB.

        If points transform as: p' = T.apply(p),
        and self.R maps box->world, then new orientation is:
            R' = T.rotation * self.R
        """
        new_center = T.apply(self.center)          # (3,)
        new_R = T.rotation * self.R                # compose rotations
        return OBB(
            center=new_center,
            half_extents=self.half_extents.copy(),
            R=new_R,
            object_id=self.object_id,
            label=self.label,
            volume=self.volume,                     # unchanged for rigid transforms
        )

    def transform_inplace(self, T: RigidTransform) -> None:
        """In-place version of transformed()."""
        self.center = T.apply(self.center)
        self.R = T.rotation * self.R
    

def points_in_obb(points: np.ndarray, obb: OBB):
    p = points - obb.center
    out = (obb.R.as_matrix().T @ p.T).T
    return (np.abs(out[:, 0]) <= obb.half_extents[0] + 1e-2) & (np.abs(out[:, 1]) <= obb.half_extents[1] + 1e-2) & (np.abs(out[:, 2]) <= obb.half_extents[2] + 1e-2)


def obb_to_o3d(obb: OBB, color: tuple[float, float, float]):
    import open3d as o3d
    # Open3D expects extent (full lengths), our half_extents are half-lengths.
    extent = (2.0 * obb.half_extents).astype(float)
    o3d_obb = o3d.geometry.OrientedBoundingBox(obb.center.astype(float), obb.R.astype(float), extent)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_obb)
    ls.paint_uniform_color(color)
    return ls
