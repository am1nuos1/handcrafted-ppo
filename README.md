# Handcrafted-PPO

Handcrafted PPO (Proximal Policy Optimization) in PyTorch.

## Goals

- Implement the essential parts of PPO by hand
- Clarify how the policy network, value network, and rollout memory work together
- Train and evaluate on Gymnasium environments

## Project Structure

```text
.
├── actor.py       # Policy network (outputs action distribution)
├── critic.py      # Value network (estimates state value)
├── memory.py      # PPO replay/memory buffer
├── agent.py       # PPO agent wrapper
├── main.py        # Training script (CartPole)
├── play.py        # Load checkpoints and watch/record evaluation
├── checkpoints/   # Saved weights (actor_torch_ppo / critic_torch_ppo)
└── cuda_test.py   # PyTorch CUDA availability check
```

## Environment & Dependencies

- Python 3.9+ (Conda/virtualenv recommended)
- Required: `torch`, `gymnasium`, `numpy`
- Watch window requires: `pygame` (install via `gymnasium[classic-control]`)
- Recording MP4 requires: `imageio`, `imageio-ffmpeg`

Install example (PowerShell):

```powershell
pip install torch gymnasium numpy
pip install "gymnasium[classic-control]"   # installs pygame for human rendering
pip install imageio imageio-ffmpeg         # recording support (mp4)
```

## Training (CartPole)

Run [main.py](main.py) to train. The best model will be saved under `checkpoints/`:

```powershell
python main.py
```

Key stats during training:
- `score`: undiscounted return per episode (CartPole gives +1 each step; equals steps survived)
- `avg_score`: moving average over the last 100 episodes
- When `avg_score` improves, `agent.save_models()` overwrites the best checkpoints

## Evaluation: Watch or Record

Use [play.py](play.py) for evaluation.

- Watch (human window, requires `pygame`):
	```powershell
	python play.py --env CartPole-v1 --episodes 3
	```

- Record (save to MP4, no popup window):
	- Save to default `videos/`:
		```powershell
		python play.py --env CartPole-v1 --episodes 3 --record
		```
	- Save to a custom directory (e.g., user-specified `vidieo/`):
		```powershell
		python play.py --env CartPole-v1 --episodes 3 --record --video-dir vidieo --video-prefix ppo-play
		```

Notes:
- [play.py](play.py) automatically loads the policy from `checkpoints/actor_torch_ppo`; it will prompt you to train first if missing.
- Watch mode uses `render_mode="human"`; record mode uses `rgb_array` with Gymnasium's `RecordVideo` to export MP4.
- Doing watch-and-record simultaneously requires extra handling; if you need it, we can add a lightweight live preview using `cv2.imshow`/`pygame`.

## Troubleshooting

- `pygame is not installed` when watching:
	```powershell
	pip install "gymnasium[classic-control]"
	```
- Missing encoder when recording:
	```powershell
	pip install imageio imageio-ffmpeg
	```
- Check CUDA availability:
	```powershell
	python cuda_test.py
	```
