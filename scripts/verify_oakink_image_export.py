"""Verify exported OakInk-Image pkl files.

Two levels of checking:

  Level 1 (always runs):
    - Tensor shapes and dtypes
    - objRot is a valid SO(3) rotation matrix
    - handTrans / objTrans are finite and have reasonable depth (Z > 0)

  Level 2 (requires --data_dir, compares against original annotations):
    - handPose   vs. re-derived from general_info (should be bit-exact)
    - handBeta   vs. general_info
    - handTrans  vs. joints_3d[0]
    - objRot / objTrans vs. obj_transf

Coordinate frame note
---------------------
All exported values are in **camera frame** (of the specific view):
  - handPose wrist (aa[0:3])  : extr_R applied, i.e. annotation→camera transform
  - handTrans                 : joints_3d[wrist] which is already camera-space
  - objRot, objTrans          : from obj_transf, which places canonical mesh in camera space
  - Finger joints (aa[3:])    : parent-relative, no global frame change needed

Usage
-----
  # Level 1 only
  python scripts/verify_oakink_image_export.py --export_dir /path/to/output

  # Level 1 + 2
  python scripts/verify_oakink_image_export.py \
      --export_dir /path/to/output \
      --data_dir   /path/to/OAKINK_DIR \
      --n_samples  20
"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot
from tqdm import tqdm

CENTER_IDX = 0

# ---------------------------------------------------------------------------
# Same rotation helpers as the export script
# ---------------------------------------------------------------------------

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _quat_wxyz_to_rotmat(q):
    q = _to_numpy(q)
    q_xyzw = q[..., [1, 2, 3, 0]]
    shape = q.shape[:-1]
    rot = SciRot.from_quat(q_xyzw.reshape(-1, 4))
    return rot.as_matrix().reshape(*shape, 3, 3)


def _rotmat_to_aa(R):
    R = _to_numpy(R)
    shape = R.shape[:-2]
    rot = SciRot.from_matrix(R.reshape(-1, 3, 3))
    return rot.as_rotvec().reshape(*shape, 3)


def _quat_wxyz_to_aa(q):
    q = _to_numpy(q)
    q_xyzw = q[..., [1, 2, 3, 0]]
    shape = q.shape[:-1]
    rot = SciRot.from_quat(q_xyzw.reshape(-1, 4))
    return rot.as_rotvec().reshape(*shape, 3)


# ---------------------------------------------------------------------------
# Level 1 checks
# ---------------------------------------------------------------------------

EXPECTED_SHAPES = {
    "handBeta":  (10,),
    "handTrans_right": (3,),
    "handPose_right":  (48,),
}


def check_shapes(frame_data, frame_path):
    errors = []
    for key, expected in EXPECTED_SHAPES.items():
        if key == "handBeta":
            t = frame_data.get("handBeta")
        elif key == "handTrans_right":
            t = frame_data.get("handTrans", {}).get("right")
        elif key == "handPose_right":
            t = frame_data.get("handPose", {}).get("right")
        else:
            t = None

        if t is None:
            errors.append(f"  MISSING key {key}")
            continue
        if not isinstance(t, torch.Tensor):
            errors.append(f"  {key}: expected torch.Tensor, got {type(t)}")
            continue
        if tuple(t.shape) != expected:
            errors.append(f"  {key}: shape {tuple(t.shape)} != expected {expected}")

    for sub_key in ("objTrans", "objRot"):
        d = frame_data.get(sub_key, {})
        if not d:
            errors.append(f"  MISSING key {sub_key}")
            continue
        for obj_id, t in d.items():
            if not isinstance(t, torch.Tensor):
                errors.append(f"  {sub_key}[{obj_id}]: not a Tensor")
                continue
            exp = (3,) if sub_key == "objTrans" else (3, 3)
            if tuple(t.shape) != exp:
                errors.append(f"  {sub_key}[{obj_id}]: shape {tuple(t.shape)} != {exp}")

    return errors


def check_geometry(frame_data):
    warnings = []

    # objRot should be SO(3)
    for obj_id, R in frame_data.get("objRot", {}).items():
        R_np = _to_numpy(R)
        det = np.linalg.det(R_np)
        ortho_err = np.max(np.abs(R_np.T @ R_np - np.eye(3)))
        if abs(det - 1.0) > 1e-4:
            warnings.append(f"  objRot[{obj_id}] det={det:.6f} (expected 1.0)")
        if ortho_err > 1e-4:
            warnings.append(f"  objRot[{obj_id}] R.T@R deviation={ortho_err:.2e}")

    # handTrans Z should be positive (object in front of camera)
    ht = frame_data.get("handTrans", {}).get("right")
    if ht is not None:
        z = float(ht[2])
        if z <= 0:
            warnings.append(f"  handTrans Z={z:.4f} (expected > 0, camera frame)")

    # objTrans Z should be positive
    for obj_id, t in frame_data.get("objTrans", {}).items():
        z = float(t[2])
        if z <= 0:
            warnings.append(f"  objTrans[{obj_id}] Z={z:.4f} (expected > 0)")

    return warnings


# ---------------------------------------------------------------------------
# Level 2: comparison against original annotations
# ---------------------------------------------------------------------------

def _recompute_from_original(anno_dir, info_str):
    """Re-derive handPose, handBeta, handTrans, objRot, objTrans from raw pkls."""
    gi_path = os.path.join(anno_dir, "general_info", f"{info_str}.pkl")
    with open(gi_path, "rb") as f:
        gi = pickle.load(f)

    hand_anno = gi["hand_anno"]
    cam_extr = _to_numpy(gi["cam_extr"])
    extr_R = cam_extr[:3, :3]

    raw_pose = _to_numpy(hand_anno["hand_pose"]).reshape(16, 4)
    wrist_q  = raw_pose[0]
    remain_q = raw_pose[1:]

    wrist_R   = extr_R @ _quat_wxyz_to_rotmat(wrist_q)
    wrist_aa  = _rotmat_to_aa(wrist_R)
    remain_aa = _quat_wxyz_to_aa(remain_q)
    hand_pose = np.concatenate([wrist_aa[None], remain_aa], axis=0).reshape(48).astype(np.float32)
    hand_shape = _to_numpy(hand_anno["hand_shape"]).astype(np.float32)

    hj_path = os.path.join(anno_dir, "hand_j", f"{info_str}.pkl")
    with open(hj_path, "rb") as f:
        joints_3d = _to_numpy(pickle.load(f))
    hand_tsl = joints_3d[CENTER_IDX].astype(np.float32)

    ot_path = os.path.join(anno_dir, "obj_transf", f"{info_str}.pkl")
    with open(ot_path, "rb") as f:
        obj_transf = _to_numpy(pickle.load(f)).astype(np.float32)
    obj_rot = obj_transf[:3, :3]
    obj_tsl = obj_transf[:3, 3]

    return hand_pose, hand_shape, hand_tsl, obj_rot, obj_tsl


def compare_with_original(frame_data, anno_dir, info_str, atol=1e-5):
    ref_pose, ref_shape, ref_tsl, ref_rot, ref_obj_tsl = _recompute_from_original(anno_dir, info_str)

    mismatches = []

    def _cmp(name, exported, reference):
        exp_np = _to_numpy(exported).astype(np.float32)
        max_err = np.max(np.abs(exp_np - reference))
        if max_err > atol:
            mismatches.append(f"  {name}: max_err={max_err:.2e}")

    _cmp("handPose", frame_data["handPose"]["right"], ref_pose)
    _cmp("handBeta", frame_data["handBeta"], ref_shape)
    _cmp("handTrans.right", frame_data["handTrans"]["right"], ref_tsl)

    for obj_id in frame_data["objRot"]:
        _cmp(f"objRot[{obj_id}]",   frame_data["objRot"][obj_id],   ref_rot)
        _cmp(f"objTrans[{obj_id}]", frame_data["objTrans"][obj_id], ref_obj_tsl)

    return mismatches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"Export dir : {args.export_dir}")
    print(f"Data dir   : {args.data_dir or '(not provided — Level 2 skipped)'}")
    anno_dir = os.path.join(args.data_dir, "image", "anno") if args.data_dir else None

    # Collect all sequence directories (those containing meta.json)
    seq_dirs = []
    for name in sorted(os.listdir(args.export_dir)):
        path = os.path.join(args.export_dir, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "meta.json")):
            seq_dirs.append(path)

    if not seq_dirs:
        print("No sequence directories with meta.json found. "
              "Re-run export_oakink_image.py to generate them.")
        return

    print(f"\nFound {len(seq_dirs)} sequence directories.")

    # Collect (seq_dir, frame_idx, info_str) tuples for sampling
    all_frames = []
    for seq_dir in seq_dirs:
        meta = json.load(open(os.path.join(seq_dir, "meta.json")))
        for i, istr in enumerate(meta["frame_info_strs"]):
            all_frames.append((seq_dir, i, istr))

    print(f"Total frames: {len(all_frames)}")

    sample_size = min(args.n_samples, len(all_frames))
    samples = random.sample(all_frames, sample_size)
    print(f"Sampling {sample_size} frames for verification.\n")

    shape_errors = 0
    geo_warnings = 0
    cmp_mismatches = 0

    for seq_dir, frame_idx, info_str in tqdm(samples, desc="Verifying"):
        pkl_path = os.path.join(seq_dir, f"{frame_idx:04d}.pkl")
        with open(pkl_path, "rb") as f:
            frame_data = pickle.load(f)

        # Level 1: shapes
        errs = check_shapes(frame_data, pkl_path)
        if errs:
            shape_errors += 1
            print(f"\n[SHAPE ERROR] {pkl_path}")
            for e in errs:
                print(e)

        # Level 1: geometry
        warns = check_geometry(frame_data)
        if warns:
            geo_warnings += 1
            print(f"\n[GEO WARNING] {pkl_path}")
            for w in warns:
                print(w)

        # Level 2: comparison
        if anno_dir:
            gi_path = os.path.join(anno_dir, "general_info", f"{info_str}.pkl")
            if not os.path.exists(gi_path):
                tqdm.write(f"  skip comparison (file not found): {gi_path}")
                continue
            mismatches = compare_with_original(frame_data, anno_dir, info_str)
            if mismatches:
                cmp_mismatches += 1
                print(f"\n[MISMATCH] {pkl_path}")
                for m in mismatches:
                    print(m)

    print("\n" + "=" * 60)
    print(f"Sampled     : {sample_size} frames")
    print(f"Shape errors: {shape_errors}")
    print(f"Geo warnings: {geo_warnings}")
    if anno_dir:
        print(f"Mismatches vs. original: {cmp_mismatches}")

    print("\nCoordinate frame summary:")
    print("  All exported values are in CAMERA frame of the annotated view.")
    print("  handPose[0:3]  = wrist axis-angle (cam_extr applied to annotation space)")
    print("  handPose[3:48] = finger joints, parent-relative (no global frame change)")
    print("  handTrans      = wrist joint 3D position (joints_3d[0], camera space)")
    print("  objRot/objTrans= SE3 from canonical (bbox-centred) mesh to camera space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify exported OakInk-Image pkl files")
    parser.add_argument("--export_dir", type=str, required=True,
                        help="directory produced by export_oakink_image.py")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="OAKINK_DIR (enables Level 2 comparison against original)")
    parser.add_argument("--n_samples", type=int, default=50,
                        help="number of frames to sample for verification")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    main(args)
