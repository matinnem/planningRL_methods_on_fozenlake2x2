# Value Iteration on 2×2 FrozenLake (Planning)

A minimal, dependency-light implementation of **value iteration** — a
dynamic-programming *planning* algorithm from reinforcement learning —
applied to a hand-coded 2×2 FrozenLake grid. No Gym / Gymnasium required.

---

## Why "planning" and not "learning"?

Value iteration assumes we already have a **perfect model** of the
environment: the transition function `T(s, a) → s'` and the reward
function `R(s, a, s')`. Given that model, it computes the optimal value
function by repeatedly applying the Bellman optimality operator:

    V_{k+1}(s) = max_a [ R(s, a, s') + γ · V_k(s') ]

That's *planning*. Reinforcement *learning* (Q-learning, SARSA, PPO…)
is when the model is unknown and has to be sampled.

---

## Environment
s1  | s2      Actions: 1 = Up 2 = Right
----+----     3 = Down 4 = Left 5 = Stay
s3  | s4


| Transition       | Reward |
|------------------|--------|
| any → `s4`       | +1     |
| any → `s2`       | −1     |
| stay (action 5)  | 0      |
| bump into wall   | −1     |
| other valid move | 0      |

---

## Algorithm

1. Initialise `V(s) = 0` for every state.
2. Repeat until `max_s |V_new(s) − V(s)| < θ`:
   - for each state, compute `Q(s, a) = R + γ·V(s')` for every action
   - set `V(s) = max_a Q(s, a)`
3. Extract the greedy policy: `π(s) = argmax_a Q(s, a)`.

Hyperparameters: `γ = 0.9`, `θ = 1e-10`.

---

## Results
Optimal State Values
s1 : 9.00000
s2 : 10.00000
s3 : 10.00000
s4 : 10.00000

Optimal Policy
s1 → Down
s2 → Down
s3 → Right
s4 → Down (Stay-in-goal loop)



> ⚠️ Note: in this simplified version `s4` is **not terminal**, so the
> agent can keep collecting `+1` forever and `V(s4) = 1 / (1 − γ) = 10`.
> The standard FrozenLake makes the goal and holes terminal; that is a
> natural next experiment.

---

## Run it

```bash
python Planning-value-iteration.py

