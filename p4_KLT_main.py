import cv2
import sys
import numpy as np
from tqdm import tqdm
from sfm_helpers import est_matr
from scipy.linalg import cholesky as cholesky2

# params for ShiTomasi corner detection
maxco = 600
feature_params = dict(maxCorners=maxco,
                      qualityLevel=0.05,
                      minDistance=5,
                      blockSize=5)

lk_params = dict(winSize=(8, 8),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class PointStream:
    def __init__(self, p_init):
        self.track = [p_init]

    def poll_prior(self):
        return self.track[-1]

    def dist_to(self, p_next):
        return np.linalg.norm(self.track[-1] - p_next)

    def push_track(self, p_next):
        self.track.append(p_next)


# Take first frame and find corners in it
old_frame = cv2.imread(f"hotel/hotel.seq0.png")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
pstream = [PointStream(p0[i, 0, :]) for i in range(0, p0.shape[0])]

color = np.random.randint(0, 255, (p0.shape[0], 3))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

t_i = 1
t_f = 40

for i in tqdm(range(t_i, t_f), desc="constructing feature tracks via OF"):
    frame = cv2.imread(f"hotel/hotel.seq{i}.png")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    a = (st == 1).squeeze()
    color = color[a]
    rstream = []
    for (keep, b) in zip(a, pstream):
        if keep:
            rstream.append(b)
    pstream = rstream
    # draw the tracks
    for e, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[e].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[e].tolist(), -1)
        pstream[e].push_track(np.asarray([c, d]))

    img = cv2.add(frame, mask)
    #    cv2.imshow('frame', img)
    #    k = cv2.waitKey(1000) & 0xff
    #    if k == 27:
    #        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    # cv2.waitKey(100)

for i in range(t_i, t_f):
    to_color = []
    im = cv2.imread(f"hotel/hotel.seq{i}.png")
    for ps in pstream:
        point = ps.track[i]
        ptc = (int((point[0] / im.shape[0]) * 255), int((point[1] / im.shape[1]) * 255), 120)
        # ptc = (255, 0, 0)
        to_color.append(ptc)
        cv2.circle(im, (point[0], point[1]), 3, ptc, -1)
    cv2.imshow("xd", im)
    cv2.waitKey(100)
cv2.imwrite("last_klt_frame.png", im)
cv2.destroyAllWindows()

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


with open("out_klt.ply", "w") as mkf:
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
