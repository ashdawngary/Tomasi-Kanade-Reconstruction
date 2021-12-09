import cv2
import numpy as np

im = cv2.imread("../hotel/hotel.seq0.png")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 1000, 0.05, 3)
corners = np.int0(corners)

t_i = 1
t_f = 40

for i in range(t_i, t_f):
    # print("frame: %s, tracks: %s" % (i, len(pstream)))
    im = cv2.imread(f"hotel/hotel.seq{i}.png")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.05, 3)
    print(corners.shape)
    corners = np.int0(corners)

    nps = []

    for i in corners:
        x, y = i.ravel()
        ptc = (int((x / im.shape[0]) * 255), int((y / im.shape[1]) * 255), 120)

        cv2.circle(im, (x, y), 3, ptc, -1)
    cv2.imshow("xd", im)
    cv2.waitKey(10)
cv2.waitKey(0)
cv2.destroyAllWindows()