"""
Policy Iteration on a 2x2 FrozenLake grid (Planning, no Gym).
============================================================
second state is a pit and the fourth state is the goal.
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
num_to_direc = {1: "↑", 2: "→", 3: "↓", 4: "←", 5: "○"}


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


# ---------------------------------------------------------------- pretty print
def render_box(v_pi, pi, title=None):
    """Print the 2x2 grid as an aligned box (fixed-width cells)."""
    W = 17
    if title:
        bar = "─" * max(1, 55 - len(title))
        print(f"\n  ── {title} {bar}")
    top = "  ╔" + "═" * W + "╦" + "═" * W + "╗"
    mid = "  ╠" + "═" * W + "╬" + "═" * W + "╣"
    bot = "  ╚" + "═" * W + "╩" + "═" * W + "╝"

    def cell(name, s):
        v = float(v_pi[s])
        a = num_to_direc[pi[s][0]]
        return [
            f"{name:^{W}}",
            f"{('v = ' + format(v, '>6.2f')):^{W}}",
            f"{a:^{W}}",
        ]

    c11, c12 = cell('s1', 's1'), cell('s2', 's2')
    c21, c22 = cell('s3', 's3'), cell('s4', 's4')

    print(top)
    for l, r in zip(c11, c12):
        print(f"  ║{l}║{r}║")
    print(mid)
    for l, r in zip(c21, c22):
        print(f"  ║{l}║{r}║")
    print(bot)


# ---------------------------------------------------------------- algorithm
pi_s = {'s1': [5,1], 's2': [5,1], 's3': [5,1], 's4': [1,1]}  # initial guess for the policy

diff1 = 1000  # for v_pi
diff_vector1 = {}
k = 0
gamma = 0.9
v_pi_s = {key: 0.0 for key in pi_s}

print("═" * 60)
print("  Policy Iteration  •  FrozenLake 2×2  (Planning)")
print("═" * 60)
print(f"  Discount factor γ   : {gamma}")
print(f"  Convergence Δ       : 1e-10")
print(f"  Total iterations    : (pending)")

render_box(v_pi_s, pi_s, title=f"Initial guess  (k = {k})")

while diff1 >= 1e-10:
    # ---- policy evaluation -------------------------------------------------
    v_pi_js = {key: 0.0 for key in pi_s}
    diff2 = 1000  # for v_pi^(j)
    diff_vector2 = {}
    j = 0
    while diff2 >= 1e-10:
        for s in state_space:
            a = pi_s[s][0]
            s2, r = transition(s, a)
            v_temp = pi_s[s][1] * (r + 0.9 * v_pi_js[s2])
            diff_vector2[s] = (float(np.abs(v_pi_js[s] - v_temp)))
            v_pi_js[s] = v_temp
            j += 1
        diff2 = max(list(diff_vector2.values()))
    diff_vector1 = [np.abs(v_pi_s[state] - v_pi_js[state]) for state in state_space]
    v_pi_s = v_pi_js.copy()
    diff1 = np.max(diff_vector1)
    k += 1

    # ---- policy improvement ------------------------------------------------
    for i, s in enumerate(state_space):
        q_values = []
        for a in action_space:
            s2, _ = transition(s, a)
            q_value = float(q(s, a, v_pi_s[s2], 0.9))
            q_values.append(q_value)
        a_max = np.argmax(q_values) + 1
        pi_s[s][0] = int(a_max)

    print(f"\n  ▸ Policy evaluation finished  (inner sweeps j = {j})")
    render_box(v_pi_s, pi_s, title=f"After policy improvement  (k = {k})")


print()
print("═" * 60)
print(f"  Converged after {k} policy-iteration steps")
print("═" * 60)
render_box(v_pi_s, pi_s, title="Final optimal policy & state values")
print()






