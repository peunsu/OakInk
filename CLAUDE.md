# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OakInk is a research dataset toolkit (`oikit`) for hand-object interaction, presented at CVPR 2022. It provides data loaders, splits, and visualization tools for three dataset components:
- **OakBase**: 1,800 household objects with part-level segmentation and affordance labels
- **OakInk-Image**: 230K multi-view images with 3D hand-object pose/shape annotations
- **OakInk-Shape**: 50K 3D hand-object grasping poses with mesh models

## Environment Setup

```bash
# Stand-alone environment
conda env create -f environment.yaml
conda activate oakink
pip install -r requirements.txt

# Or install as package into existing env
pip install .
```

**Required**: Set `OAKINK_DIR` environment variable to the dataset root path before using the toolkit.

## Key Scripts

```bash
# Visualize OakInk-Image frames
python scripts/viz_oakink_image.py --data_split train --mode_split subject

# Sequence-level visualization (wireframe or mesh mode)
python scripts/viz_oakink_image_seq.py --draw_mode mesh --view_id 1

# Visualize OakInk-Shape grasping poses
python scripts/viz_oakink_shape.py --categories teapot --intent_mode use

# OakBase demo
python scripts/demo_oak_base.py

# Dataset utilities
python scripts/unzip_all.py      # Extract dataset archives (requires 7zip)
python scripts/verify_checksum.py  # Validate downloaded files via SHA256
```

## Package Architecture

```
oikit/
├── common.py          # Rotation conversion utilities (axis-angle, quat, rotmat, 6D, Euler)
├── oak_base.py        # OakBase: affordance knowledge, part segmentation, 29 attribute labels
├── oi_image/
│   ├── oi_image.py    # OakInkImage: frame-level loader with 3 split modes
│   ├── oi_image_mv.py # OakInkImageSequence: sequence/multi-view handler with MANO
│   ├── utils.py       # Projection, mesh loading utilities
│   └── viz_tool.py    # OpenDR-based mesh renderer
└── oi_shape/
    ├── oi_shape.py    # OakInkShape: grasping pose loader with MD5 caching
    └── utils.py       # Metadata and object loading utilities
```

### Core Classes

**OakBase** — affordance knowledge base:
```python
oak = OakBase()
objs = oak.get_objs_by_category("teapot")
objs = oak.get_objs_by_attribute("observe_sth")  # 29 semantic attribute labels
obj.get_part_name_by_attribute("observe_sth")
obj.part_name_to_segs[part_name]  # Returns PLY file path
```

**OakInkImage** — frame-level dataset (232K samples):
- 3 split modes: `subject` (SP1), `object` (SP2), `view` (SP0)
- Splits: `train`, `val`, `test`, `all`
```python
oi = OakInkImage(data_split="train", mode_split="subject")
oi.get_image(idx)
oi.get_joints_2d(idx), oi.get_joints_3d(idx)
oi.get_corners_2d(idx)
```

**OakInkImageSequence** — sequence-level with MANO hand model:
- Sequence ID format: `A_B_C` (obj/intent/subject) or `A_B_C_D` (handover with giver/receiver)
```python
seq = OakInkImageSequence(seq_id="A01001_0001_0000/2021-09-26-19-59-58", view_id=0)
seq.get_mano_pose(i), seq.get_mano_shape(i)  # 16 quat params, 10 shape params
seq.get_verts_3d(i), seq.get_obj_verts_3d(i)
seq.get_hand_over(i)  # Handover sequences only
```

**OakInkShape** — 3D grasping poses (train: 49,302 / val: 6,522 / test: 6,222):
- 34 object categories, 4 intent modes: `use`, `hold`, `liftup`, `handover`
- MD5-based caching for fast reload
```python
shape = OakInkShape(data_split="train", category="teapot", intent_mode="use")
shape[idx]  # Returns grasp data dict
```

### Rotation Utilities (`oikit.common`)

All conversions use PyTorch3D conventions. Available: `aa_to_rotmat`, `quat_to_aa`, `rotmat_to_ee`, `rot6d_to_rotmat`, and inverses.

## Dataset Structure

```
$OAKINK_DIR/
├── OakBase/
│   └── {category}/{obj_id}/
│       ├── part_*.ply         # Part segmentation point clouds
│       └── part_*.json        # Part affordance attributes
├── image/
│   ├── anno/
│   │   ├── general_info/      # 3D hand/object pose per frame
│   │   ├── hand_j/, hand_v/   # 21 joints, 778 MANO vertices
│   │   ├── obj_trasnf/        # Object transforms
│   │   └── split*/            # SP0/SP1/SP2 split files
│   ├── obj/                   # Object meshes (.obj files)
│   └── stream_release_v2/     # Raw images (4 camera views)
└── shape/
    ├── metaV2/                # Category and object ID metadata
    ├── oakink_shape_v2/       # Grasping pose data
    ├── OakInkObjectsV2/       # Source object meshes
    └── OakInkVirtualObjectsV2/  # Target object meshes for transfer
```

Intent ID mapping: `0001=use`, `0002=hold`, `0003=liftup`, `0004=handover`

## Key Dependencies

- **3D/Vision**: PyTorch3D, trimesh, open3d, OpenDR (rendering)
- **Hand Model**: `manotorch` (custom MANO layer, MANO pose in quaternion format)
- **Transforms**: transforms3d
