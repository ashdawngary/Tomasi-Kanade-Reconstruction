import numpy as np
from typing import List, Tuple
import sympy


def est_matr(corre: List[Tuple[np.ndarray, np.ndarray]]):
    A = np.zeros((3 * len(corre), 6)).astype(np.float64)
    b = np.zeros((3 * len(corre), 1)).astype(np.float64)

    for (ci, (vec_left, vec_right)) in enumerate(corre):
        (x, y, z) = vec_left
        (xp, yp, zp) = vec_right
        A[3 * ci, :] = [x * x, 2 * x * y, 2 * x * z, y * y, 2 * z * y, z * z]
        A[3 * ci + 1, :] = [xp * xp, yp * xp + xp * yp, zp * xp + xp * zp, yp * yp, yp * zp + zp * yp, zp * zp]
        A[3 * ci + 2, :] = [x * xp, y * xp + x * yp, z * xp + x * zp, y * yp, y * zp + z * yp, z * zp]

        b[3 * ci + 0, 0] = 1
        b[3 * ci + 1, 0] = 1
        b[3 * ci + 2, 0] = 0

    lsq, resid, rank, sing = np.linalg.lstsq(A, b, rcond=None)
    lsq = lsq.squeeze()
    # for i in range(0, 3 * len(corre)):
    #    print("check %s: %s => %s" % (i, np.matmul(A[i, :], lsq), b[i]))

    return np.asarray([[lsq[0], lsq[1], lsq[2]],
                       [lsq[1], lsq[3], lsq[4]],
                       [lsq[2], lsq[4], lsq[5]]])





if __name__ == '__main__':
    # limit test l m a o
    i1 = np.asarray([3, 1, 0])
    i2 = np.asarray([-2, 4, -1])
    i3 = np.asarray([2.5, 2, 4])
    j1 = np.asarray([5, 1, 4])
    j2 = np.asarray([2, 2, 2])
    j3 = np.asarray([-1, 5, 4])
    pairs = [
        (i1.T, j1.T),
        (i2.T, j2.T)
        # (i3.T, j3.T)
    ]
    A = np.asarray([[0.02465504720406704, 0.10733478576615763, -0.08780864197530891],
                    [0.10733478576615763, 0.13409586056644868, -0.07722403776325265],
                    [-0.08780864197530891, -0.07722403776325265, 0.2066448801742924]])

    # theorized lsq sol

    C = est_matr(pairs)  # should be p e r f e c t
    print(C)

    for pair in pairs:
        (i, j) = pair
        print(i, j)
        print(np.matmul(i.T, np.matmul(C, i)),
              np.matmul(j.T, np.matmul(C, j)),
              np.matmul(i.T, np.matmul(C, j)))
