import cv2
import numpy as np

# params for ShiTomasi corner detection
maxco = 600
feature_params = dict(maxCorners=maxco,
                      qualityLevel=0.05,
                      minDistance=5,
                      blockSize=5)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(8, 8),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors

# Take first frame and find corners in it
old_frame = cv2.imread(f"hotel/hotel.seq0.png")

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

color = np.random.randint(0, 255, (p0.shape[0], 3))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

t_i = 1
t_f = 40

for i in range(t_i, t_f):
    frame = cv2.imread(f"hotel/hotel.seq{i}.png")
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    a = (st == 1).squeeze()
    color = color[a]
    # draw the tracks
    for e, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[e].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[e].tolist(), -1)
    img = cv2.add(frame, mask)
    cv2.imshow('frame', img)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    # cv2.waitKey(100)

print(p0.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()
