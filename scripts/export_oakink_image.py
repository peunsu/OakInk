"""Export OakInk-Image data as frame-by-frame pkl files in WORLD coordinate space.

Requires: numpy, torch, scipy, tqdm.

Output layout:
  {output_dir}/
    {seq_dir}/
      0000.pkl, 0001.pkl, ...
    obj/
      {obj_id}.obj

Each pkl is a dict:
  handBeta              : (10,)     torch.Tensor  — MANO shape params
  handTrans  > right    : (3,)      torch.Tensor  — wrist joint position (world)
  handPose   > right    : (48,)     torch.Tensor  — 16 joints in axis-angle (world)
  handKeypoints > right : (21, 3)   torch.Tensor  — 21 hand joint positions (world)
  handVertex    > right : (778, 3)  torch.Tensor  — MANO mesh vertices (world)
  objTrans   > OBJ_ID   : (3,)      torch.Tensor  — object translation (world)
  objRot     > OBJ_ID   : (3,3)     torch.Tensor  — object rotation matrix (world)

Notes:
  - handTrans is the wrist joint position in world space: camera-space joints_3d[0]
    (from hand_j) transformed by inv(cam_extr).
  - handPose[0:3] is the global wrist axis-angle in world space. The raw MANO
    quaternion in hand_anno is stored in world space; cam_extr is NOT applied
    (contrast with export_oakink_image.py which converts to camera space).
  - handPose[3:] are finger relative rotations (identical to camera-space export).
  - objRot / objTrans come from general_info["obj_anno"] (T_w_o, world←object
    canonical). Falls back to converting obj_transf via inv(cam_extr) when
    obj_anno is absent.
"""

import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_INTENT = {
    "use": "0001",
    "hold": "0002",
    "liftup": "0003",
    "handover": "0004",
}

SPLIT_KEY_MAP = {
    "default": "split0",
    "subject": "split1",
    "object": "split2",
}

CENTER_IDX = 0  # wrist joint used as hand translation origin


# ---------------------------------------------------------------------------
# Rotation utilities (scipy-based, no pytorch3d required)
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_wxyz_to_aa(q):
    """(…, 4) [w,x,y,z] → (…, 3)"""
    q = _to_numpy(q)
    q_xyzw = q[..., [1, 2, 3, 0]]
    shape = q.shape[:-1]
    rot = SciRot.from_quat(q_xyzw.reshape(-1, 4))
    return rot.as_rotvec().reshape(*shape, 3)


# ---------------------------------------------------------------------------
# Info list loading
# ---------------------------------------------------------------------------

def _info_str(info):
    s = "__".join([str(x) for x in info])
    return s.replace("/", "__")


def _load_info_list(data_dir, data_split, mode_split):
    anno_dir = os.path.join(data_dir, "image", "anno")
    split_key = SPLIT_KEY_MAP.get(mode_split, "split0")

    if data_split == "all":
        return json.load(open(os.path.join(anno_dir, "seq_all.json")))
    if data_split == "train+val":
        return json.load(open(os.path.join(anno_dir, "split", split_key, "seq_train.json")))
    if data_split == "train":
        return json.load(open(
            os.path.join(anno_dir, "split_train_val", split_key, "example_split_train.json")))
    if data_split == "val":
        return json.load(open(
            os.path.join(anno_dir, "split_train_val", split_key, "example_split_val.json")))
    return json.load(open(os.path.join(anno_dir, "split", split_key, "seq_test.json")))


# ---------------------------------------------------------------------------
# Per-frame parsing — WORLD coordinate space
# ---------------------------------------------------------------------------

def _parse_frame(anno_dir, info):
    """Return (hand_pose_48, hand_shape_10, hand_tsl_3, joints_world_21x3, obj_rot_33, obj_tsl_3, obj_id)
    all in world coordinate space."""
    istr = _info_str(info)

    # ---- general_info ----
    gi_path = os.path.join(anno_dir, "general_info", f"{istr}.pkl")
    with open(gi_path, "rb") as f:
        gi = pickle.load(f)

    hand_anno = gi["hand_anno"]
    cam_extr = _to_numpy(gi["cam_extr"])  # (4, 4) world→camera (T_c_w)
    T_w_c = np.linalg.inv(cam_extr)       # (4, 4) camera→world

    # ---- MANO pose: raw quaternions are in world space ----
    raw_pose = _to_numpy(hand_anno["hand_pose"]).reshape(16, 4)  # (16,4) [w,x,y,z]
    wrist_q  = raw_pose[0]    # (4,) world-space global orientation
    remain_q = raw_pose[1:]   # (15,4) relative finger rotations (frame-independent)

    # Wrist: convert raw world-space quaternion to axis-angle — no cam_extr needed
    wrist_aa  = _quat_wxyz_to_aa(wrist_q)   # (3,) world space
    remain_aa = _quat_wxyz_to_aa(remain_q)  # (15,3)

    hand_pose  = np.concatenate([wrist_aa[None], remain_aa], axis=0).reshape(48).astype(np.float32)
    hand_shape = _to_numpy(hand_anno["hand_shape"]).astype(np.float32)  # (10,)

    # ---- hand joints: camera→world ----
    hj_path = os.path.join(anno_dir, "hand_j", f"{istr}.pkl")
    with open(hj_path, "rb") as f:
        joints_cam = _to_numpy(pickle.load(f))  # (21, 3) camera space

    joints_h = np.concatenate([joints_cam, np.ones((21, 1))], axis=1)  # (21, 4)
    joints_world = (T_w_c @ joints_h.T).T[:, :3].astype(np.float32)   # (21, 3) world space
    hand_tsl = joints_world[CENTER_IDX]                                  # (3,) wrist

    # ---- hand vertices: camera→world ----
    hv_path = os.path.join(anno_dir, "hand_v", f"{istr}.pkl")
    with open(hv_path, "rb") as f:
        verts_cam = _to_numpy(pickle.load(f))  # (778, 3) camera space

    verts_h = np.concatenate([verts_cam, np.ones((778, 1))], axis=1)  # (778, 4)
    verts_world = (T_w_c @ verts_h.T).T[:, :3].astype(np.float32)    # (778, 3) world space

    # ---- object transform: world space ----
    if "obj_anno" in gi:
        # T_w_o stored directly in general_info
        obj_anno = _to_numpy(gi["obj_anno"]).astype(np.float32)  # (4,4)
    else:
        # Fallback: load camera-space obj_transf and convert to world space
        ot_path = os.path.join(anno_dir, "obj_transf", f"{istr}.pkl")
        with open(ot_path, "rb") as f:
            T_c_o = _to_numpy(pickle.load(f)).astype(np.float32)  # (4,4) cam←obj
        obj_anno = (T_w_c @ T_c_o).astype(np.float32)             # (4,4) world←obj

    obj_rot = obj_anno[:3, :3]   # (3,3)
    obj_tsl = obj_anno[:3, 3]    # (3,)

    # ---- obj_id ----
    seq_cat = info[0].split("/")[0]
    obj_id = seq_cat.split("_")[0]

    return hand_pose, hand_shape, hand_tsl, joints_world, verts_world, obj_rot, obj_tsl, obj_id


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(args):
    os.makedirs(args.output_dir, exist_ok=True)
    obj_out_dir = os.path.join(args.output_dir, "obj")
    os.makedirs(obj_out_dir, exist_ok=True)

    anno_dir = os.path.join(args.data_dir, "image", "anno")
    obj_src_dir = os.path.join(args.data_dir, "image", "obj")

    info_list = _load_info_list(args.data_dir, args.data_split, args.mode_split)
    print(f"Loaded {len(info_list)} frame entries")

    if args.view_id is not None:
        info_list = [info for info in info_list if info[3] == args.view_id]
        print(f"  → {len(info_list)} after view_id={args.view_id} filter")

    if args.intent_mode != "all":
        target_intent = ALL_INTENT[args.intent_mode]
        info_list = [info for info in info_list
                     if info[0].split("/")[0].split("_")[1] == target_intent]
        print(f"  → {len(info_list)} after intent_mode={args.intent_mode} filter")

    seq_groups = defaultdict(list)
    for info in info_list:
        seq_key = (info[0], info[1], info[3])
        seq_groups[seq_key].append(info)
    for key in seq_groups:
        seq_groups[key].sort(key=lambda x: x[2])

    print(f"  → {len(seq_groups)} sequences to export\n")

    exported_obj_ids = set()
    total_frames = 0

    for seq_key, seq_frames in tqdm(seq_groups.items(), desc="Sequences"):
        seq_id, sub_id, view_id = seq_key
        seq_cat = seq_id.split("/")[0]
        obj_id = seq_cat.split("_")[0]
        action_id = seq_cat.split("_")[1]

        safe_seq = seq_id.replace("/", "__")
        if action_id == "0004":
            seq_dir_name = f"{safe_seq}__sub{sub_id}__view{view_id}"
        else:
            seq_dir_name = f"{safe_seq}__view{view_id}"
        seq_out_dir = os.path.join(args.output_dir, seq_dir_name)
        os.makedirs(seq_out_dir, exist_ok=True)

        frame_info_strs = []
        frame_ids = []

        for frame_idx, info in enumerate(seq_frames):
            try:
                hand_pose, hand_shape, hand_tsl, joints_world, verts_world, obj_rot, obj_tsl, _ = _parse_frame(anno_dir, info)
            except FileNotFoundError as e:
                tqdm.write(f"  skip {_info_str(info)}: {e}")
                continue

            frame_data = {
                "handBeta":      torch.from_numpy(hand_shape),
                "handTrans":     {"right": torch.from_numpy(hand_tsl)},
                "handPose":      {"right": torch.from_numpy(hand_pose)},
                "handKeypoints": {"right": torch.from_numpy(joints_world)},
                "handVertex":    {"right": torch.from_numpy(verts_world)},
                "objTrans":      {obj_id: torch.from_numpy(obj_tsl)},
                "objRot":        {obj_id: torch.from_numpy(obj_rot)},
            }

            frame_path = os.path.join(seq_out_dir, f"{frame_idx:04d}.pkl")
            with open(frame_path, "wb") as f:
                pickle.dump(frame_data, f)
            total_frames += 1
            frame_info_strs.append(_info_str(info))
            frame_ids.append(info[2])

        meta = {
            "seq_id": seq_id,
            "obj_id": obj_id,
            "sub_id": sub_id,
            "view_id": view_id,
            "frame_ids": frame_ids,
            "frame_info_strs": frame_info_strs,
        }
        with open(os.path.join(seq_out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if obj_id not in exported_obj_ids:
            for ext in (".obj", ".ply"):
                src = os.path.join(obj_src_dir, f"{obj_id}{ext}")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(obj_out_dir, f"{obj_id}{ext}"))
                    exported_obj_ids.add(obj_id)
                    break
            else:
                tqdm.write(f"  warning: no mesh found for obj_id={obj_id}")

    print(f"\nDone. {total_frames} frames → {args.output_dir}")
    print(f"      {len(exported_obj_ids)} object meshes → {obj_out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export OakInk-Image data as frame-by-frame pkl files (world coordinate space)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="path to OAKINK_DIR (dataset root)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="output directory")
    parser.add_argument("--data_split", type=str, default="train",
                        choices=["all", "train+val", "train", "val", "test"])
    parser.add_argument("--mode_split", type=str, default="default",
                        choices=["default", "subject", "object"])
    parser.add_argument("--view_id", type=int, default=None, choices=[0, 1, 2, 3],
                        help="camera view (0-3); omit to export all views")
    parser.add_argument("--intent_mode", type=str, default="all",
                        choices=list(ALL_INTENT) + ["all"])
    args = parser.parse_args()
    export(args)
