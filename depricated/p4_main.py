import random

import cv2
import numpy as np
import sys
from tqdm import tqdm
from sfm_helpers import est_matr
from scipy.linalg import cholesky as cholesky2


class PointStream:
    def __init__(self, p_init):
        self.track = [p_init]

    def poll_prior(self):
        return self.track[-1]

    def dist_to(self, p_next):
        return np.linalg.norm(self.track[-1] - p_next)

    def push_track(self, p_next):
        self.track.append(p_next)


im = cv2.imread("../hotel/hotel.seq0.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 600, 0.01, 3)
corners = np.int0(corners)

pstream = []
for corner in corners:
    x, y = corner.ravel()
    pstream.append(PointStream(np.asarray([x, y])))

t_i = 1
t_f = 40

for i in tqdm(range(t_i, t_f), desc="computing feature paths"):
    # print("frame: %s, tracks: %s" % (i, len(pstream)))
    im = cv2.imread(f"hotel/hotel.seq{i}.png")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 600, 0.01, 3)
    corners = np.int0(corners)

    nps = []

    for i in corners:
        x, y = i.ravel()
        psn = np.asarray([x, y])
        cv2.circle(im, (x, y), 3, 255, -1)
        det = None
        for ps in pstream:
            if ps.dist_to(psn) <= 1.5:
                det = ps

        if det is not None:
            det.push_track(psn)
            pstream.remove(det)
            nps.append(det)

    pstream = nps  # recycle

to_color = []  # colormaps for correlation!

for i in range(t_i, t_f):
    to_color = []
    im = cv2.imread(f"hotel/hotel.seq{i}.png")
    for ps in pstream:
        point = ps.track[i]
        ptc = (int((point[0] / im.shape[0])*255), int((point[1] / im.shape[1])*255), 120)
        #ptc = (255, 0, 0)
        to_color.append(ptc)
        cv2.circle(im, (point[0], point[1]), 3, ptc, -1)
    cv2.imshow("xd", im)
    cv2.waitKey(10)

cv2.imwrite("lastframe.png", im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# construct w matrix
centroids = []

frames = len(pstream[0].track)
W = np.zeros((2 * frames, len(pstream)))

for (psix, ps) in enumerate(pstream):
    for (trackix, v) in enumerate(ps.track):
        W[trackix, psix] = v[0]  # y
        W[trackix + frames, psix] = v[1]  # x

W = W - np.mean(W, axis=1, keepdims=True)
np.set_printoptions(threshold=sys.maxsize)

u, d, vt = np.linalg.svd(W)

up = u[:, :3]
dp = d[:3]
vp = vt[:3, :]
M = np.matmul(up, np.diag(np.sqrt(dp)))

S = np.matmul(np.diag(np.sqrt(dp)), vp)

ij_pairs = []

for pi in range(0, len(pstream[0].track)):
    ij_pairs.append((np.transpose(M[pi, :]),
                     np.transpose(M[pi + frames, :])))

L = est_matr(ij_pairs)

'''
for pair in ij_pairs:
    (i, j) = pair
    print(i, j)
    print(np.matmul(i.T, np.matmul(L, i)),
          np.matmul(j.T, np.matmul(L, j)),
          np.matmul(i.T, np.matmul(L, j)))
'''

C = cholesky2(L)

M = np.matmul(M, C)
S = np.matmul(np.linalg.inv(C), S)

Wk = np.matmul(M, S)

W_residual = W - Wk


def closest(sources, sourceix):
    pts = []
    src = sources[:, sourceix]
    for a in range(0, sources.shape[1]):
        p2 = sources[:, a]
        if sourceix != a:
            pts.append((a, np.linalg.norm(p2 - src)))

    return min(pts, key=lambda e: e[1])[0]


with open("out.ply", "w") as mkf:
    mkf.write("ply\nformat ascii 1.0"
              + "\nelement vertex %s" % S.shape[1]
              + "\nproperty float x"
              + "\nproperty float y"
              + "\nproperty float z"
                "\nproperty uchar red"
                "\nproperty uchar green"
                "\nproperty uchar blue"
                "\nelement edge %s" % S.shape[1]
              + "\nproperty int vertex1"
                "\nproperty int vertex2"
                "\nend_header\n")
    for vecix in range(0, S.shape[1]):
        p = S[:, vecix]
        ptc = to_color[vecix]
        mkf.write("%s %s %s %s %s %s\n" % (
            p[1], p[0], p[2], ptc[2], ptc[1], ptc[0]))  # possible bgr moment

    for vecix in range(0, S.shape[1]):
        mkf.write("%s %s\n" % (vecix, closest(S, vecix)))
    mkf.close()
