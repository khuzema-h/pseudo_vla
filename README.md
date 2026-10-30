# pseudo-vla

A single-process Isaac Sim loop that finds an object, tracks it, and moves a Franka arm to it.

The environment camera finds the object. [SAM 2](https://github.com/facebookresearch/sam2) masks it on the first frame, [CoTracker3](https://github.com/facebookresearch/co-tracker) follows points on that mask, and the depth image turns those points into a 3D goal. [cuRobo](https://nvlabs.github.io/curobo) 0.8 plans a collision-free joint path, and the arm follows it. The session stays open so you can send another object or send the arm home.

This was brought up on an RTX 3080 Laptop (16 GB) with Python 3.12 and Torch 2.6.0+cu124. Leave Torch on that build. Isaac Sim 6.2's own pin is newer, and upgrading Torch breaks this stack.

## What is in the scene

Isaac Sim's simple room, with the Franka standing on the near edge of the table. Three 5 cm dynamic cubes sit in front of it: a red block, a green block, and a blue block.

Two cameras:

- **Wrist camera.** Mounted on the gripper and aimed into the fingers. The Isaac window titled "Wrist Camera" shows this view.
- **Environment camera.** Fixed in front of the arm. Perception uses this image. The window titled "SAM + CoTracker" shows the mask and the tracked points.

cuRobo treats the table and all three blocks as cuboid obstacles, including the block being approached.

## Layout

| Path | Role |
| --- | --- |
| `scripts/track_and_servo.py` | Interactive entry point |
| `pseudo_vla/scene.py` | Room, Franka, colored blocks, and both cameras |
| `pseudo_vla/perception.py` | OWL-ViT grounding, SAM 2, CoTracker3, depth unprojection |
| `pseudo_vla/servo.py` | cuRobo planner and joint playback |
| `pseudo_vla/view.py` | Wrist viewport and the SAM + CoTracker window |
| `checkpoints/` | SAM 2 weights (not committed) |
| `segment-anything-2/` | Local SAM 2 clone, installed editable (not committed) |

## Install

Create the conda env from the repo root:

```bash
conda env create -f environment.yml
conda activate ps_vla
```

Install Torch 2.6 for CUDA 12.4 before anything else:

```bash
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then the rest of the Python dependencies. `requirements.txt` lists Isaac Sim 6.2, CoTracker, and Transformers. If a later install upgrades Torch, pin it back to `2.6.0+cu124`.

```bash
pip install -r requirements.txt
pip install 'nvidia-curobo==0.8.0.post1.dev43' --extra-index-url https://pypi.nvidia.com
```

`cuda-core[cu12]` is in `requirements.txt`. cuRobo needs that CUDA 12 backend; without it, `MotionPlannerCfg.create` fails with "No curobo kernel backend available".

Clone SAM 2 into this repo and install it editable. Keep Torch pinned if that install tries to change it.

```bash
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
SAM2_BUILD_CUDA=1 pip install -e segment-anything-2
pip install torch==2.6.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Download the SAM 2 base-plus checkpoint and the CoTracker3 online checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
mkdir -p ~/.cache/torch/hub/checkpoints
wget -O ~/.cache/torch/hub/checkpoints/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

The first run downloads `google/owlvit-base-patch32` into the Hugging Face cache. Isaac Sim downloads the simple room and Franka from its asset server on first launch, so that run needs network access.

Check that the GPU stack imports:

```bash
python -c "import torch, curobo, cotracker, sam2; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Run

From the repo root, with the `ps_vla` env active and a display available:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
python scripts/track_and_servo.py
```

The first Kit launch is slow. Later launches are much faster. When the prompt appears:

- Type a description, such as `red block`, `green block`, or `blue block`. OWL-ViT finds a box and SAM 2 segments it.
- Type two numbers for a pixel on the environment image, or four numbers for a box (`x1 y1 x2 y2`).
- Type `home` to plan back to the pose the hand had at startup.
- Type `quit` to close Kit.

A move ends when the gripper is centered over the selected block and within 2 cm of the 10 cm hover waypoint. The sim keeps running after that. Typing another command during a move interrupts it.

Useful flags:

```bash
python scripts/track_and_servo.py --target "red block"       # first prompt, then stay open
python scripts/track_and_servo.py --headless                  # no Isaac windows
python scripts/track_and_servo.py --max-steps 600             # steps allowed for one move
```

Logs for each step go to `outputs/run.log`. The first segmented frame is saved to `outputs/frame0_overlay.png`.

## Current behavior

- One process shares the GPU across Isaac Sim, SAM 2, CoTracker3, OWL-ViT, and cuRobo.
- SAM 2 and OWL-ViT load once when perception starts and stay resident on the GPU between prompts.
- Perception reads the environment camera. The wrist camera is a view of the gripper, not the finder.
- Tracked points are unprojected with the depth image. The approach pose is 10 cm above the median point, with the hand pointing down.
- The arm replans when the 3D goal moves. All blocks, including the current target, plus the table, are collision cuboids in the robot base frame.
- The gripper does not close. The motion stops above the object.
- The scene provides movable blocks, but the current controller does not pick up or stack them.
