# Demos

## Demo 0 — Watch the pipeline overview

The repo includes a screen recording of the lite COLMAP / ORB workflow:

**[`lite_colmap_demo.mp4`](../lite_colmap_demo.mp4)** (~2.5 min, 1080p)

Download or clone the full repository to view locally. GitHub may not inline-play large MP4 files in the README.

---

## Demo 1 — Inspect golden output (no build required)

Browse the precomputed workspace under **`Models/Full model example/`**:

| File | Description |
|---|---|
| `outpy.h264` | Input video |
| `images/` | 87 extracted frames |
| `camera_poses/` | Fixed poses for `point_triangulator` |
| `database.db` | ORB features in COLMAP DB format |
| `points3D.txt`, `images.txt` | Sparse reconstruction |
| `pointData.csv`, `descriptorsData.xml`, `sparse.xyz` | RBD export formats |
| `sfm_log.txt` | Stage timings (~71 s total on reference hardware) |

Use this folder to understand expected outputs before running on your own video.

---

## Demo 2 — Run on bundled sample video

```bash
# After building COLMAP + Ceres (see install.rst)
pip install -r requirements.txt

# One-time: generate ref_camera_pose.h5 (from Orb_version/)
cd Orb_version
python3 camera_pose_preparation.py -p ../Models/video1

# Production run
mkdir -p /tmp/my_workspace
cp Models/video1/outpy.h264 /tmp/my_workspace/
./colmap_bash.sh -p /tmp/my_workspace
```

Sample inputs also live in `Models/video2/` and `Models/video3/`.

---

## Demo 3 — Exit-point only (post-reconstruction)

If you already have `points3D.txt` and `images.txt` at the workspace root:

```bash
cd Orb_version
python3 Find_exit/find_room_exit.py --path /path/to/workspace
# → exit_point.csv
```

The golden example in `Models/Full model example/` can be used if those files are present at the workspace root.

---

## Demo 4 — Full COLMAP path (non-ORB)

Edit `Orb_version/colmap_bash.sh`:

```bash
use_orb_version=false
```

Then run `./colmap_bash.sh -p "$WORKSPACE"`. Uses `colmap automatic_reconstructor --quality=extreme` instead of the ORB Python matcher. Slower on weak CPUs but no `ref_camera_pose.h5` dependency.

---

## Demo 5 — Raspberry Pi capture (optional)

Uncomment the `scan_script.py` block in `colmap_bash.sh` and run on a Pi with Tello + PiCamera:

```bash
python3 scan_script.py --output "$WORKSPACE"
```

Requires `picamera` and `djitellopy` (see `requirements.txt` comments).
