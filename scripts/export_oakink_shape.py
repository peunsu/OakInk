"""Export OakInk-Shape grasps as frame-by-frame pkl files.

Standalone — no oikit / manotorch dependency required.
Only needs: numpy, torch, tqdm (all standard ML deps).
"""

import argparse
import glob
import hashlib
import json
import os
import pickle
import re
import shutil

import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (from oikit/oi_shape/utils.py)
# ---------------------------------------------------------------------------

ALL_CAT = [
    "apple", "banana", "binoculars", "bottle", "bowl", "cameras", "can", "cup",
    "cylinder_bottle", "donut", "eyeglasses", "flashlight", "fryingpan",
    "gamecontroller", "hammer", "headphones", "knife", "lightbulb", "lotion_pump",
    "mouse", "mug", "pen", "phone", "pincer", "power_drill", "scissors",
    "screwdriver", "squeezable", "stapler", "teapot", "toothbrush",
    "trigger_sprayer", "wineglass", "wrench",
]

ALL_SPLIT = ["train", "val", "test"]

ALL_INTENT = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}


# ---------------------------------------------------------------------------
# Utilities (inlined from oikit/oi_shape/utils.py)
# ---------------------------------------------------------------------------

def _get_obj_path(oid, data_path, meta_path, use_downsample=False):
    obj_suffix_path = "align_ds" if use_downsample else "align"
    real_meta = json.load(open(os.path.join(meta_path, "object_id.json"), "r"))
    virtual_meta = json.load(open(os.path.join(meta_path, "virtual_object_id.json"), "r"))
    if oid in real_meta:
        obj_name = real_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkObjectsV2")
    else:
        obj_name = virtual_meta[oid]["name"]
        obj_path = os.path.join(data_path, "OakInkVirtualObjectsV2")
    candidates = (
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.obj")) +
        glob.glob(os.path.join(obj_path, obj_name, obj_suffix_path, "*.ply"))
    )
    if len(candidates) > 1:
        candidates = [p for p in candidates if "align" in os.path.basename(p)]
    assert len(candidates) == 1, f"Expected exactly one mesh for {oid}, got: {candidates}"
    return candidates[0]


def _split_for_obj_id(oid):
    """Returns 'train', 'val', or 'test' based on MD5 hash of obj_id."""
    h = int(hashlib.md5(oid.encode("utf-8")).hexdigest(), 16)
    r = h % 10
    if r < 8:
        return "train"
    elif r == 8:
        return "val"
    else:
        return "test"


# ---------------------------------------------------------------------------
# Grasp list builder (inlined from OakInkShape._prepare_data)
# ---------------------------------------------------------------------------

def _build_grasp_list(oi_shape_dir, categories, intent_idx_set, data_splits):
    seq_cat_matcher = re.compile(r"(.+)/(.{6})_(.{4})_([_0-9]+)/([\-0-9]+)")
    grasp_list = []

    for cat in tqdm(categories, desc="Scanning categories"):
        real_matcher = re.compile(rf"({re.escape(cat)}/(.{{6}})/.{{10}})/hand_param\.pkl$")
        virtual_matcher = re.compile(rf"({re.escape(cat)}/(.{{6}})/.{{10}})/(.{{6}})/hand_param\.pkl$")
        cat_path = os.path.join(oi_shape_dir, cat)
        if not os.path.isdir(cat_path):
            continue

        for cur, dirs, files in os.walk(cat_path, followlinks=False):
            dirs.sort()
            for fname in files:
                full = os.path.join(cur, fname)
                rel = os.path.relpath(full, oi_shape_dir)

                vm = virtual_matcher.findall(rel)
                is_virtual = len(vm) > 0
                rm = real_matcher.findall(rel)
                matches = vm if is_virtual else rm
                if not matches:
                    continue

                assert len(matches) == 1
                m = matches[0]
                # m for real:    (path_prefix, raw_oid)
                # m for virtual: (path_prefix, raw_oid, virtual_oid)

                source_path = os.path.join(oi_shape_dir, m[0], "source.txt")
                source = open(source_path).read().strip()
                gc = seq_cat_matcher.findall(source)
                if not gc:
                    continue
                gc = gc[0]
                pass_stage, raw_obj_id, action_id, subject_id, seq_ts = (
                    gc[0], gc[1], gc[2], gc[3], gc[4]
                )

                obj_id = m[2] if is_virtual else m[1]

                if action_id not in intent_idx_set:
                    continue
                if _split_for_obj_id(obj_id) not in data_splits:
                    continue

                grasp_list.append({
                    "obj_id": obj_id,
                    "file_path": full,
                    "action_id": action_id,
                    "subject_id": subject_id.split("_")[0] if action_id == "0004" else subject_id,
                    "seq_ts": seq_ts,
                    "source": source,
                    "pass_stage": pass_stage,
                    "is_virtual": is_virtual,
                    "raw_obj_id": raw_obj_id,
                })

    return grasp_list


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(args):
    os.makedirs(args.output_dir, exist_ok=True)
    obj_dir = os.path.join(args.output_dir, "obj")
    os.makedirs(obj_dir, exist_ok=True)

    categories = ALL_CAT if args.category == "all" else [args.category]
    data_splits = set(ALL_SPLIT if args.data_split == "all" else [args.data_split])
    intent_modes = list(ALL_INTENT.keys()) if args.intent_mode == "all" else [args.intent_mode]
    intent_idx_set = {ALL_INTENT[m] for m in intent_modes}

    data_dir = os.path.join(args.data_dir, "shape")
    oi_shape_dir = os.path.join(data_dir, "oakink_shape_v2")
    meta_dir = os.path.join(data_dir, "metaV2")

    grasp_list = _build_grasp_list(oi_shape_dir, categories, intent_idx_set, data_splits)
    print(f"Total grasps: {len(grasp_list)}")

    seq_frame_counters = {}
    exported_obj_ids = set()

    for grasp in tqdm(grasp_list, desc="Exporting frames"):
        with open(grasp["file_path"], "rb") as f:
            raw_param = pickle.load(f)

        hand_pose = torch.from_numpy(raw_param["pose"].astype(np.float32)).reshape(48)  # (48,)
        hand_shape = torch.from_numpy(raw_param["shape"].astype(np.float32))            # (10,)
        hand_tsl = torch.from_numpy(raw_param["tsl"].astype(np.float32))               # (3,)

        obj_id = grasp["obj_id"]

        frame_data = {
            "handBeta": hand_shape,
            "handTrans": {"right": hand_tsl},
            "handPose": {"right": hand_pose},
            "objTrans": {obj_id: torch.zeros(3)},
            "objRot": {obj_id: torch.eye(3)},
        }

        seq_dir = os.path.join(args.output_dir, obj_id)
        os.makedirs(seq_dir, exist_ok=True)
        frame_idx = seq_frame_counters.get(obj_id, 0)
        frame_path = os.path.join(seq_dir, f"{frame_idx:04d}.pkl")
        with open(frame_path, "wb") as f:
            pickle.dump(frame_data, f)
        seq_frame_counters[obj_id] = frame_idx + 1

        if obj_id not in exported_obj_ids:
            src_path = _get_obj_path(obj_id, data_dir, meta_dir, use_downsample=False)
            dst_path = os.path.join(obj_dir, f"{obj_id}.obj")
            shutil.copy2(src_path, dst_path)
            exported_obj_ids.add(obj_id)

    print(f"Exported {len(grasp_list)} frames across {len(seq_frame_counters)} sequences to {args.output_dir}")
    print(f"Exported {len(exported_obj_ids)} obj meshes to {obj_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OakInk-Shape grasps as frame-by-frame pkl files")
    parser.add_argument("--data_dir", type=str, default="data", help="OAKINK_DIR path")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--category", type=str, default="all", help="object category or 'all'")
    parser.add_argument("--intent_mode", type=str, default="all",
                        choices=list(ALL_INTENT) + ["all"], help="intent mode or 'all'")
    parser.add_argument("--data_split", type=str, default="train",
                        choices=ALL_SPLIT + ["all"], help="data split or 'all'")
    args = parser.parse_args()
    export(args)
