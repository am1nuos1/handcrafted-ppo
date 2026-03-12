# Handcrafted-PPO

A hand-crafted PPO (Proximal Policy Optimization) practice project built with PyTorch.

## Goals

- Implement the core components of PPO manually
- Understand how policy networks, value networks, and rollout memory are organized
- Build a foundation for integrating Gym / Gymnasium environments and a full training loop later

## Project Structure

```text
.
├── actor.py      # Policy network that outputs an action distribution
├── critic.py     # Value network that estimates state values
├── memory.py     # Sample buffer used during PPO training
├── agent.py      # PPO agent wrapper
└── cuda_test.py  # Simple CUDA availability check for PyTorch
```
