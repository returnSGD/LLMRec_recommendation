"""
Training script for the RL + Memory recommendation system.

Two-phase training:
  Phase 1: Behavior Cloning (BC) warm-start
    - Collect (state, action) pairs from P5's recommendations
    - Train policy to mimic P5's implicit "strategy" (i.e., which items it recommends)
    - Provides a reasonable initialization before RL

  Phase 2: Offline RL (CQL or IQL)
    - Use pre-collected interaction data as the offline dataset
    - Train Q-network with conservative / implicit regularization
    - Fine-tune policy with advantage-weighted regression (IQL) or
      direct Q-maximization (CQL)

Usage:
  python -m src.train --config configs/beauty_cql.yaml
"""
import os
import sys
import argparse
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from .config import Config, MemoryConfig, RLConfig, TrainingConfig, RewardWeights
from .memory import MemoryManager, MemoryEncoder, MemoryEncoderPretrained
from .rl import (
    POMDPEnv, POMDPPolicy, RewardFunction, UncertaintyAdaptiveEpsilon,
    ReplayBuffer, EpisodicReplayBuffer,
    CQLTrainer, IQLTrainer, ValueNetwork,
    ACTION_REGISTRY, NUM_ACTIONS, action_to_candidate_strategy,
)
from .model import MemoryRLModel


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: Config, p5_model=None) -> MemoryRLModel:
    """Build the full RL + Memory model from config."""
    return MemoryRLModel(config, p5_model=p5_model)


def collect_offline_data(env: POMDPEnv,
                         model: MemoryRLModel,
                         data_loader,
                         num_episodes: int = 1000) -> ReplayBuffer:
    """
    Phase 0: Collect interaction data for offline RL.

    Uses a mix of the current policy (ε-greedy) and P5's recommendations
    to build a diverse offline dataset.
    """
    buffer = ReplayBuffer(capacity=500_000)

    for episode_idx in range(num_episodes):
        # Get a user from the data loader
        batch = next(iter(data_loader))
        user_ids = batch.get("user_ids", [episode_idx])
        user_embs = batch.get("user_embs", np.random.randn(len(user_ids), 256))

        for i, user_id in enumerate(user_ids):
            user_emb = user_embs[i].numpy() if isinstance(user_embs, torch.Tensor) else user_embs[i]
            state = env.reset(user_id=int(user_id), initial_user_emb=user_emb)

            for step in range(env.max_steps):
                # Decide action using current policy
                result = model.forward(user_emb, user_id=int(user_id))
                action_id = result["action_id"]

                # Simulate user feedback (placeholder — replace with real data)
                feedback = _simulate_feedback(action_id, env)

                next_state, reward, done, info = env.step(action_id, feedback)
                buffer.add(state, action_id, reward, next_state, done, info)
                state = next_state

                if done:
                    break

    return buffer


def _simulate_feedback(action_id: int, env: POMDPEnv) -> Dict:
    """Placeholder: simulate user feedback. Replace with real user model."""
    action = ACTION_REGISTRY[action_id]
    return {
        "clicked": random.random() < 0.3,
        "purchased": random.random() < 0.05,
        "dwell_time": random.uniform(0, 30),
        "new_categories": 1 if action.is_exploration else 0,
        "negative_signal": "skip" if random.random() < 0.35 else None,
        "retention": 1.0,
        "item_id": random.randint(0, 10000),
        "rating": random.uniform(3, 5),
        "category_id": random.randint(0, 50),
        "recommended_categories": {random.randint(0, 50)},
    }


def train_behavior_cloning(model: MemoryRLModel,
                           buffer: ReplayBuffer,
                           num_steps: int = 5000,
                           batch_size: int = 256,
                           lr: float = 1e-3,
                           device: str = "cuda"):
    """
    Phase 1: Behavior Cloning — train policy to mimic dataset actions.
    """
    model.policy.to(device)
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=lr)
    model._train_step = 0

    for step in range(num_steps):
        if len(buffer) < batch_size:
            continue

        batch = buffer.sample(batch_size, device)
        q_values = model.policy(
            batch["user_emb"], batch["memory_context"],
            batch["retrieval_confidence"], batch["periodic_flags"],
            batch["action_mask"],
        )

        bc_loss = nn.CrossEntropyLoss()(q_values, batch["actions"])

        optimizer.zero_grad()
        bc_loss.backward()
        optimizer.step()
        model._train_step += 1

        if step % 500 == 0:
            acc = (q_values.argmax(-1) == batch["actions"]).float().mean()
            print(f"BC Step {step}: loss={bc_loss.item():.4f}, acc={acc.item():.3f}")

    print("Behavior Cloning complete.")


def train_offline_rl(model: MemoryRLModel,
                     buffer: ReplayBuffer,
                     config: Config,
                     device: str = "cuda"):
    """
    Phase 2: Offline RL training using CQL or IQL.
    """
    writer = SummaryWriter(log_dir=config.training.log_dir)

    if config.rl.algorithm == "cql":
        print("Training with CQL...")
        target_policy = POMDPPolicy(
            user_dim=model.user_dim,
            memory_dim=config.memory.memory_dim,
            num_actions=NUM_ACTIONS,
            hidden_dim=config.rl.hidden_dim,
            num_layers=config.rl.num_layers,
            dropout=config.rl.dropout,
        )

        trainer = CQLTrainer(
            policy=model.policy,
            target_policy=target_policy,
            buffer=buffer,
            device=device,
            gamma=config.rl.gamma,
            tau=config.rl.tau,
            cql_alpha=config.rl.cql_alpha,
            cql_num_samples=config.rl.cql_num_samples,
            lr=config.rl.lr,
            grad_clip=config.rl.grad_clip,
            writer=writer,
        )

    elif config.rl.algorithm == "iql":
        print("Training with IQL...")
        v_net = ValueNetwork(
            user_dim=model.user_dim,
            memory_dim=config.memory.memory_dim,
            hidden_dim=config.rl.hidden_dim,
        )

        trainer = IQLTrainer(
            q_network=model.policy,
            v_network=v_net,
            buffer=buffer,
            device=device,
            gamma=config.rl.gamma,
            tau=config.rl.tau,
            expectile=config.rl.iql_expectile,
            temperature=config.rl.iql_temperature,
            lr=config.rl.lr,
            grad_clip=config.rl.grad_clip,
            writer=writer,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.rl.algorithm}")

    # Training loop
    trainer.train(
        num_steps=config.training.total_steps,
        batch_size=config.rl.batch_size,
        eval_every=config.training.eval_every,
    )

    # Save
    os.makedirs(config.training.output_dir, exist_ok=True)
    trainer.save(os.path.join(config.training.output_dir, "rl_checkpoint.pt"))

    writer.close()
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="cql", choices=["cql", "iql"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p5_checkpoint", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    # Build config
    config = Config()
    config.rl.algorithm = args.algorithm

    # Load P5 model if available
    p5_model = None
    if args.p5_checkpoint and os.path.exists(args.p5_checkpoint):
        print(f"Loading P5 checkpoint from {args.p5_checkpoint}")
        # Placeholder: load actual P5 model
        # from modeling_p5 import P5
        # p5_model = P5.from_pretrained(...)
        pass

    # Build model
    model = build_model(config, p5_model=p5_model)
    print(f"Model built. User dim: {model.user_dim}, Memory dim: {config.memory.memory_dim}")
    print(f"RL algorithm: {config.rl.algorithm}, Actions: {NUM_ACTIONS}")
    print(f"Action space: {[a.name for a in ACTION_REGISTRY.values()]}")

    # Build environment
    env = POMDPEnv(
        memory_manager=model.memory_manager,
        user_encoder=model.encode_user,
        reward_function=model.reward_fn,
    )

    # Collect offline data
    print("\nPhase 0: Collecting offline data...")
    buffer = collect_offline_data(env, model, data_loader=None, num_episodes=100)
    print(f"Collected {len(buffer)} transitions.")

    # Phase 1: Behavior Cloning warm-start
    print("\nPhase 1: Behavior Cloning warm-start...")
    train_behavior_cloning(model, buffer, num_steps=2000, device=args.device)

    # Phase 2: Offline RL
    print("\nPhase 2: Offline RL training...")
    train_offline_rl(model, buffer, config, device=args.device)

    print("\nTraining complete!")
    print(f"Model saved to {config.training.output_dir}")


if __name__ == "__main__":
    main()
