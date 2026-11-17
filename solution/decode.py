"""Three-state HMM (H, A, D) decoded with Viterbi.

Score of a path = sum of per-frame evidence for the chosen states - cost of every state change.
Viterbi returns the path with the best total score, exactly (dynamic programming, O(frames x 9)).
It is the 1-D analogue of a Potts / total-variation denoiser: piecewise-constant output, sharp
boundaries, and a cost per jump that sets how many segments survive.
"""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def transition_matrix(P):
    """t[i, j] = score added when moving from state i to state j (0 to stay)."""
    t = np.zeros((3, 3))
    t[H, A] = t[A, H] = -P.SWITCH_COST
    t[H, D] = t[A, D] = -P.TO_DEAD_COST
    t[D, H] = t[D, A] = -P.FROM_DEAD_COST
    return t


def emissions(dead_e, owner_e):
    """(frames, 3) per-frame scores. Alive states share the owner evidence, D takes dead evidence."""
    e = np.zeros((len(dead_e), 3))
    e[:, H] = 0.5 * owner_e
    e[:, A] = -0.5 * owner_e
    e[:, D] = dead_e
    return e


def viterbi(emis, trans):
    """Exact Viterbi. Plain-Python scalar loop: ~10x faster than numpy for 3 states."""
    n = len(emis)
    e = emis.tolist()
    t = trans.tolist()
    back = bytearray(3 * n)
    d0, d1, d2 = e[0]
    for i in range(1, n):
        new = []
        for j in range(3):
            c0 = d0 + t[0][j]
            c1 = d1 + t[1][j]
            c2 = d2 + t[2][j]
            if c0 >= c1 and c0 >= c2:
                best, arg = c0, 0
            elif c1 >= c2:
                best, arg = c1, 1
            else:
                best, arg = c2, 2
            back[3 * i + j] = arg
            new.append(best + e[i][j])
        d0, d1, d2 = new
    last = [d0, d1, d2]
    state = max(range(3), key=lambda k: (last[k], -k))
    path = np.empty(n, dtype=np.int8)
    path[-1] = state
    for i in range(n - 1, 0, -1):
        state = back[3 * i + state]
        path[i - 1] = state
    return path
