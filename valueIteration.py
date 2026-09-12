"""
Value Iteration on a 2x2 FrozenLake grid (Planning, no Gym).
============================================================

Value Iteration is a *planning* algorithm: it assumes a perfect model of
the environment (transition + reward functions) and computes the optimal
value function and policy by dynamic programming.
"""
import numpy as np

# ---------------------------------------------------------------- environment
# 2x2 grid layout:
#     s1 | s2
#    ----+----
#     s3 | s4
#
# Action mapping (matches transition() below):
#     1 = Up    2 = Right    3 = Down    4 = Left    5 = Stay
action_space = [1, 2, 3, 4, 5]
state_space = ['s1', 's2', 's3', 's4']

def reward(state1, action1 ,state2):
    if state1 == 's4':
        if action1 == 4:
            return 0
        elif action1 == 5:
            return 1
        else:
            return -1
    elif state2 == 's4':
        return 1
    elif state2 == 's2':
        return -1
    elif state1 == state2:
        if action1 != 5:
            return -1
        else:
            return 0
    elif state1 != state2:
        return 0


def transition(state1, action1):
    if action1 == 5:
        state2 = state1
        return state2, reward(state1, action1, state2)
    else:
        if state1 == 's1':
            if action1 == 2:
                state2 = 's2'
                return state2, reward(state1, action1, state2)
            elif action1 == 3:
                state2 = 's3'
                return state2, reward(state1, action1, state2)
            else:
                state2 = state1
                return state2, reward(state1, action1, state2)
        if state1 == 's2':
            if action1 == 3:
                state2 = 's4'
                return state2, reward(state1, action1, state2)
            elif action1 == 4:
                state2 = 's1'
                return state2, reward(state1, action1, state2)
            else:
                state2 = state1
                return state2, reward(state1, action1, state2)
        if state1 == 's3':
            if action1 == 1:
                state2 = 's1'
                return state2, reward(state1, action1, state2)
            elif action1 == 2:
                state2 = 's4'
                return state2, reward(state1, action1, state2)
            else:
                state2 = state1
                return state2, reward(state1, action1, state2)
        if state1 == 's4':
            if action1 == 1:
                state2 = 's3'
                return state2, reward(state1, action1, state2)
            elif action1 == 4:
                state2 = 's2'
                return state2, reward(state1, action1, state2)
            else:
                state2 = state1
                return state2, reward(state1, action1, state2)

def q(state1, action1, v, gamma = 0.9):
    state2 , r = transition(state1, action1)
    return r + gamma * v

vs = {'s1': 0.0, 's2': 0.0, 's3': 0.0, 's4': 0.0}
vs_temp = vs.copy()
ss = vs.copy()
diff = 1000
k = 0
while diff >= 1e-10:
    for i, s in enumerate(state_space):
        q_values = []
        for a in action_space:
            s2, _ = transition(s, a)
            q_value = float(q(s, a, vs[s2], 0.9))
            q_values.append(q_value)
        a_max = np.argmax(q_values) + 1
        s_new = transition(s, a_max)
        ss[s] = s_new
        vs[s] = float(np.max(q_values))
    diff = np.sum([abs(vs[key] - vs_temp[key]) for key in vs])
    vs_temp = vs.copy()
    k += 1
# ------------------------------------------------------------------ reporting
vs = {key: round(value, 5) for key, value in vs.items()}

ACTION_NAMES = {1: "Up", 2: "Right", 3: "Down", 4: "Left", 5: "Stay"}

line = "═" * 60

print(line)
print("  FrozenLake 2×2  •  Value Iteration (Planning)")
print(line)
print(f"  Discount factor γ  : 0.9")
print(f"  Iterations         : {k}")
print()

# ---- state values with little bars ----------------------------------------
print("  Optimal State Values")
print("  " + "─" * 50)
max_v = max(vs.values()) if max(vs.values()) > 0 else 1
for s in state_space:
    v = vs[s]
    bar_len = int(round((v / max_v) * 20)) if max_v else 0
    bar = "█" * bar_len
    print(f"    {s}  →  {v:8.5f}   {bar}")
print()

# ---- policy ---------------------------------------------------------------
print("  Optimal Policy")
print("  " + "─" * 50)
print(f"    {'State':<8}{'Next State':<12}{'Reward':<8}")
print("  " + "─" * 50)
for s in state_space:
    next_state, r = ss[s]
    print(f"    {s:<8}{next_state:<12}{r:<8}")
print(line)
