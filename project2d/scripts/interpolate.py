"""
We decided to annotate occupancy over tracker data, so we want to add some `pre-annotations` 
for occupancy collected from tacker. This script does precisely that: Creates interpolates 
from tracker annotations for occupancy annotation.
"""

import os
import json
import argparse
from copy import deepcopy
from glob import glob

from tqdm import tqdm
from lib.readers import pcd_reader
from scipy.spatial.transform import Rotation
from lib.common.file_utils import _index_frames
from lib.common import box
from lib.common.label_lut import CLASS_MAPPING_tracker_to_occupancy as CLASS_MAPPING 
from lib.common.label_lut import IGNORE_CLASSES_tracker_to_occupancy as IGNORE_CLASSES

OUTPUT_DIR = "./output"
ID_INCREMENT = 10000

def remap_classes_in_objects(objects):
    """
    Mutate objects in-place: map pedestrians and cars to 'moving_now'.
    """
    objects_refined = []
    
    for obj in objects:
        # Try a few common keys; adjust if your schema is different
        cls = obj.get("obj_type")
        if cls in IGNORE_CLASSES:
            pass

        if cls in CLASS_MAPPING:
            obj["obj_type"] = CLASS_MAPPING[cls]
            objects_refined.append(obj)

    return objects

def remap_classes_in_scene(scene_dir):
    """
    For every frame_*.json in scene_dir, map pedestrian/car -> 'moving_now'
    and write back to the same file.
    """
    for path in glob(os.path.join(scene_dir, "frame_*.json")):
        with open(path, "r") as f:
            data = json.load(f)

        # assuming top-level is a list of objects
        remap_classes_in_objects(data)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)


def lerp(v0, v1, alpha):
    return v0 + alpha * (v1 - v0)

def interpolate_bboxes(psr0, psr1, alpha):
    """Linearly interpolate psr dict; if one side is missing, just copy the other."""
    if psr0 is None and psr1 is None:
        return None
    if psr0 is None:
        # object only exists in frame 1
        return deepcopy(psr1)
    if psr1 is None:
        # object only exists in frame 0
        return deepcopy(psr0)

    out = deepcopy(psr0)

    for group in ["half_extents", "center"]:
        for axis in [0, 1, 2]:
            v0 = getattr(psr0, group)[axis]
            v1 = getattr(psr1, group)[axis]
            getattr(out, group)[axis] = lerp(v0, v1, alpha)
    
    v0 = getattr(psr0, "R").as_quat()
    v1 = getattr(psr1, "R").as_quat()
    out.R = Rotation.from_quat(lerp(v0, v1, alpha))
            
    return out

def load_frames(data_dir, all=False):
    frames = {}
    paths = _index_frames(data_dir, "json")

    for idx, path in paths.items():
        if all or (not all and ((idx % 10 == 1) or (idx % 10 == 6))):
            with open(path, "r") as f:
                frames[idx] = [box.OBB.from_json_entry(item) for item in json.load(f)]
    return frames


def save_frame(data_dir, index, objects):
    out_path = os.path.join(data_dir, f"frame_{index:06d}.json")
    os.remove(out_path)
    with open(out_path, "w") as f:
        json.dump(objects, f, indent=4)


def build_obj_dict(objects):
    """Map obj_id -> object dict."""
    return {obj.object_id: obj for obj in objects}


def interpolate_between_frames(idx0, objs0, idx1, objs1):
    """
    Given:
      idx0, idx1: frame indices (ints)
      objs0, objs1: lists of objects (from JSON)
    Return:
      dict: frame_index -> list_of_objects for intermediate indices
    """
    interpolated_frames = {}
    delta = idx1 - idx0
    if delta <= 1:
        return interpolated_frames  # nothing to fill

    d0 = build_obj_dict(objs0)
    d1 = build_obj_dict(objs1)

    all_ids = set(d0.keys()) | set(d1.keys())

    # For each intermediate frame
    for k in range(idx0 + 1, idx1):
        alpha = (k - idx0) / float(delta)
        frame_objects = []

        for oid in all_ids:
            o0 = d0.get(oid)
            o1 = d1.get(oid)

            # Skip if object is missing in both (shouldn't happen)
            if o0 is None and o1 is None:
                continue

            # Choose template object (carry metadata from one of the endpoints)
            template = o0 if o0 is not None else o1
            new_obj = deepcopy(template)

            new_obj = interpolate_bboxes(o0, o1, alpha)

            frame_objects.append(new_obj.to_json())

        interpolated_frames[k] = frame_objects

    return interpolated_frames

def copy_detections_from_window(frames, transforms, center_idx, window_size, out_path):
    """
    For the given frame index `center_idx`, copy all detections from the temporal
    window [center_idx - window_size, center_idx + window_size] INTO that frame.
    """

    if center_idx not in frames:
        return

    # Start with whatever is already in the center frame
    new_frame_objects = [i.to_json() for i in frames[center_idx]]
    T_ref = transforms[center_idx]

    # Look over neighbor frames within the window
    for offset in range(-window_size, window_size):
        if offset == 0:
            continue  # keep center frame as the base
        idx = center_idx + offset
        if idx not in frames:
            continue

        # Append all objects from this neighbor frame
        for obj in frames[idx]:
            T = transforms[idx]
            obj = obj.transformed(T.inv())
            obj_ref = obj.transformed(T_ref)
            obj_json = deepcopy(obj_ref.to_json())
            obj_json["obj_id"] = str(int(obj_json["obj_id"]) + (window_size + offset + 1) * ID_INCREMENT)
            new_frame_objects.append(obj_json)

    # Update the center frame
    frames[center_idx] = new_frame_objects
    save_frame(out_path, center_idx, new_frame_objects)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="./output", help="Path to save")
    ap.add_argument("--home-folder", type=str, default="/downloaded_data/datasets/occupancy/to_merge/", help="Path to dataset with bag recordings")
    args = ap.parse_args()


    for scene_dir in tqdm(os.listdir(args.home_folder)):
        output_dir = os.path.join(args.output_dir, scene_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_labels = os.path.join(output_dir, "pre_label")
        os.system(f"cp -r {os.path.join(args.home_folder, scene_dir)}/meta {output_dir}")
        
        scene_dir = os.path.join(args.home_folder, scene_dir, "label")
        mystery_number = os.listdir(scene_dir)[0]
        
        os.makedirs(output_labels, exist_ok=True)
        os.system(f"cp -r {os.path.join(args.home_folder, scene_dir, mystery_number)}/* {output_labels}")
        
        remap_classes_in_scene(output_labels)
        
        frames = load_frames(output_labels)
        if not frames:
            print("No frames found.")

        # sort frame indices
        sorted_indices = sorted(frames.keys())

        # interpolate between each consecutive annotated pair
        for i in range(len(sorted_indices) - 1):
            idx0 = sorted_indices[i]
            idx1 = sorted_indices[i + 1]
            objs0 = frames[idx0]
            objs1 = frames[idx1]

            inter = interpolate_between_frames(idx0, objs0, idx1, objs1)
            # write missing frames
            for k, objs in inter.items():
                save_frame(output_labels, k, objs)

        frames = load_frames(output_labels, True)
        meta_paths = _index_frames(os.path.join(output_dir, "meta"), "json")

        reader = pcd_reader.PCDReader()
        transforms = {idx: reader.read_pose(path) for idx, path in meta_paths.items()}

        assert list(transforms.keys()) == list(frames.keys()), "not the same frames in meta files and annotations"

        for frame_ind in range(6, 200, 10):
            copy_detections_from_window(frames, transforms, frame_ind, 5, output_labels)
