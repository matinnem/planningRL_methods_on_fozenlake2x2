
# Policy Iteration on a 2×2 FrozenLake

A minimal, dependency-light implementation of **policy iteration** — one
of the classical *planning* algorithms of reinforcement learning —
applied to a hand-coded 2×2 FrozenLake grid.
No Gym, no Gymnasium. Only Python + NumPy.

> Companion branch: `value-iteration` for the other classical planner.

---

## Why "planning", not "learning"?

In reinforcement learning, **planning** assumes you already have a perfect
model of the environment (`R(s, a, s')` and `T(s, a) → s'`). Given the
model, you can compute the optimal value function and policy **without
interacting with the environment**.

That is different from **learning**, where the model is unknown and the
values have to be estimated from sampled experience (Q-learning, SARSA,
PPO…). This repo sits firmly on the planning side.

---

## Environment
s1 | s2 Actions: 1 = Up 2 = Right
----+---- 3 = Down 4 = Left 5 = Stay
s3 | s4



- `s2` is a pit (reward −1)
- `s4` is the goal (reward +1)
- any invalid move (bumping into a wall) gives −1
- "stay" (action 5) gives 0 unless it lands on the goal or pit

---

## Algorithm

This section presents another important algorithm: **policy iteration**.
Unlike value iteration, policy iteration is *not* for directly solving
the Bellman optimality equation. However, it has an intimate relationship
with value iteration, and the idea behind it is widely used in modern
reinforcement learning algorithms.

Policy iteration is an iterative algorithm. Each iteration has two steps:

1. **Policy evaluation.** Given a policy `π_k`, compute its state value
   `v_{π_k}` by solving the Bellman equation:
        v_{π_k} = r_{π_k} + γ · P_{π_k} · v_{π_k}

where `r_{π_k}` and `P_{π_k}` come from the system model.

2. **Policy improvement.** Using `v_{π_k}`, produce a new, better policy:
        π_{k+1} = arg max_π ( r_π + γ · P_π · v_{π_k} )

Repeat until the value (or the policy) stops changing.

### Elementwise form

**Policy evaluation** solves `v_{π_k} = r_{π_k} + γ·P_{π_k}·v_{π_k}`
iteratively, one state at a time:

v_{π_k}^{(j+1)}(s) = Σ_a π_k(a|s) ·
[ Σ_r p(r|s,a)·r

    γ · Σ_{s'} p(s'|s,a) · v_{π_k}^{(j)}(s') ]
    for all s ∈ S, j = 0, 1, 2, ...



**Policy improvement** computes, for each state,
π_{k+1}(s) = arg max_π Σ_a π(a|s) ·
( Σ_r p(r|s,a)·r + γ · Σ_{s'} p(s'|s,a) · v_{π_k}(s') )
└──────────────── q_{π_k}(s, a) ────────────────┘


Let `a*_k(s) = arg max_a q_{π_k}(s, a)`. Then the greedy policy is
π_{k+1}(a|s) = 1 if a == a*_k(s),
0 otherwise


### Connection to value iteration

Value iteration performs **one Bellman-optimality update per sweep**.
Policy iteration performs **many Bellman-expectation updates** (policy
evaluation) followed by **one greedy update** (policy improvement). Both
converge to the same optimal value function and policy — policy iteration
often in far fewer outer iterations, at the cost of the inner evaluation
loop.

---

## Results

With `γ = 0.9` and initial guess `π_0 = {stay everywhere, up in s4}`:

- Policy evaluation converges in ~880 inner sweeps per round.
- **Policy iteration converges in 3 outer iterations.**
- Final values: `s1 = 9, s2 = 10, s3 = 10, s4 = 10` — the same as
  value iteration, as expected.


Optimal Policy
s1 → Down
s2 → Down
s3 → Right
s4 → Right


---

## Run it

```bash
python policyIteration.py

Requires only Python 3.8+ and NumPy.

