import os
import argparse
import numpy as np

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from gymnasium.wrappers import RecordVideo

from agent import Agent


def play(
	env_id: str = "CartPole-v0",
	episodes: int = 3,
	render: bool = True,
	record: bool = False,
	video_dir: str = "videos",
	video_prefix: str = "ppo-play"
):
	# Build env depending on watch/record options
	# - record requires render_mode="rgb_array" for RecordVideo
	# - watch uses render_mode="human" (needs pygame installed)
	if record:
		os.makedirs(video_dir, exist_ok=True)
		try:
			env = gym.make(env_id, render_mode="rgb_array")
		except TypeError:
			env = gym.make(env_id)
		env = RecordVideo(env, video_folder=video_dir, name_prefix=video_prefix)
		# When recording, don't rely on env.render() display
		render = False
	else:
		if render:
			try:
				env = gym.make(env_id, render_mode="human")
			except TypeError:
				env = gym.make(env_id)
		else:
			env = gym.make(env_id)

	# Build agent with matching shapes and actions
	agent = Agent(
		n_actions=env.action_space.n,
		batch_size=1,
		alpha=3e-4,
		n_epochs=1,
		input_dims=env.observation_space.shape,
	)

	# Ensure checkpoint exists before loading
	actor_ckpt = os.path.join("checkpoints", "actor_torch_ppo")
	if not os.path.exists(actor_ckpt):
		print(f"Actor checkpoint not found at {actor_ckpt}. Train first via main.py.")
		return

	# Load saved weights
	agent.actor.load_checkpoint()

	# Roll out episodes
	for ep in range(episodes):
		# Some gymnasium envs attempt to render on reset() when render_mode="human"
		# Handle missing pygame gracefully by falling back to no-render mode
		try:
			obs, info = env.reset()
		except DependencyNotInstalled as e:
			if record:
				print("Video recording dependency missing:")
				print("  Try: pip install imageio imageio-ffmpeg")
			print(f"Render dependency missing ({e}). Falling back to headless run.")
			env.close()
			env = gym.make(env_id)  # no render_mode
			obs, info = env.reset()
		done = False
		ep_reward = 0.0

		while not done:
			# Choose action from current policy (stochastic sample)
			action, _, _ = agent.choose_action(obs)
			# Some environments require int actions explicitly
			action = int(action)

			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			ep_reward += float(reward)

			# For explicit render calls, ignore if rendering is unavailable
			if render:
				try:
					env.render()
				except Exception:
					pass

		print(f"Episode {ep+1}/{episodes} | return: {ep_reward:.1f}")
		if record:
			try:
				last_path = env.video_recorder.path  # type: ignore[attr-defined]
				if last_path:
					print(f"Saved video: {last_path}")
			except Exception:
				pass

	env.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Play with saved PPO actor")
	parser.add_argument("--env", dest="env_id", type=str, default="CartPole-v0",
						help="Environment id (default: CartPole-v0)")
	parser.add_argument("--episodes", type=int, default=3,
						help="Number of evaluation episodes")
	parser.add_argument("--no-render", action="store_true",
						help="Disable rendering even if available")
	parser.add_argument("--record", action="store_true",
						help="Record episodes to video (uses rgb_array)")
	parser.add_argument("--video-dir", type=str, default="videos",
						help="Directory to save videos (default: videos)")
	parser.add_argument("--video-prefix", type=str, default="ppo-play",
						help="Filename prefix for saved videos")
	args = parser.parse_args()

	play(
		env_id=args.env_id,
		episodes=args.episodes,
		render=not args.no_render,
		record=args.record,
		video_dir=args.video_dir,
		video_prefix=args.video_prefix,
	)

