# RL Planning Algorithms — 2×2 FrozenLake

Two classical **planning** algorithms from reinforcement learning,
implemented from scratch with only Python + NumPy. No Gym, no Gymnasium.

Each algorithm lives on its own branch and has its own README:

| Algorithm         | Branch             | File                  |
|-------------------|--------------------|-----------------------|
| Value Iteration   | `value-iteration`  | `valueIteration.py`   |
| Policy Iteration  | `policy-iteration` | `policyIteration.py`  |

---

## What's the difference?

**Value iteration** directly applies the Bellman *optimality* operator:
V_{k+1}(s) = max_a [ R(s, a, s') + γ · V_k(s') ]


sweeping every state until `V` stops changing. The optimal policy is then
read off greedily: `π(s) = argmax_a Q(s, a)`.

**Policy iteration** is not a direct solver of the Bellman optimality
equation. Instead it alternates between two steps:

1. **Policy evaluation** — solve `v_π = r_π + γ·P_π·v_π` for the current
   policy `π`.
2. **Policy improvement** — `π' = argmax_π ( r_π + γ·P_π·v_π )`.

Both algorithms converge to the same optimal value function and policy.
Value iteration does **one** Bellman-optimality update per sweep; policy
iteration does **many** Bellman-expectation updates (the inner evaluation)
plus **one** greedy update. The idea behind policy iteration is widely
used in modern RL.

---

## Environment (shared by both branches)

Hand-coded 2×2 FrozenLake:
s1 | s2 Actions: 1 = Up 2 = Right
----+---- 3 = Down 4 = Left 5 = Stay
s3 | s4



- `s2` = pit (reward −1)
- `s4` = goal (reward +1)
- invalid move (bumping a wall) = −1
- `γ = 0.9`

Same reward and transition functions in both branches.

---

## Results (identical optimum, as expected)

| State | Value |
|-------|-------|
| `s1`  | 9     |
| `s2`  | 10    |
| `s3`  | 10    |
| `s4`  | 10    |

Optimal policy:
s1 ↓ s2 ↓
s3 → s4 →


Number of iterations:

- Value iteration: ~170–330 outer sweeps (depends on the convergence
  threshold).
- Policy iteration: 3 outer iterations, each with ~880 inner evaluation
  sweeps.

---

## How to use this repo

```bash
# clone
git clone git@github.com:<USERNAME>/<REPO>.git
cd <REPO>

# run value iteration
git checkout value-iteration
python valueIteration.py

# run policy iteration
git checkout policy-iteration
python policyIteration.py

Requirements: Python 3.8+ and NumPy.
