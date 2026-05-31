"""
RL + Long-term Memory Recommendation System
============================================
Two core contributions:
  1. Hierarchical external memory (short-term + long-term with time decay)
  2. POMDP-based RL decision framework with uncertainty-driven exploration
Plus:
  3. Memory uncertainty → adaptive exploration strength
  4. Efficient discrete-action inference (1 forward pass vs P5's beam=20)

Entry points:
  python -m src.rl_experiment  → Comparison experiment (P5 vs RL+Memory)
"""
