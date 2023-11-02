import focal_cal_auto
import monocular_demo
import cv2
import numpy as np

monocular_demo.KNOWN_WIDTH = 11.0

monocular_demo.FOCAL_LENGTH = focal_cal_auto.get_focal_img(24.0, monocular_demo.KNOWN_WIDTH, "./focal.jpeg")

print(monocular_demo.FOCAL_LENGTH)

raw = cv2.imread("./data.jpeg")
marker = focal_cal_auto.find_marker(raw)
if marker == 0:
    print(marker)
    exit(0)

inches = monocular_demo.distance_to_camera(monocular_demo.KNOWN_WIDTH, monocular_demo.FOCAL_LENGTH, marker[1][0])

box = np.int0(cv2.boxPoints(marker))

cv2.drawContours(raw, [box], -1, (0, 255, 0), 2)

cv2.putText(raw, "%.2fcm" % (inches*30.48 / 12),
		(raw.shape[1] - 200, raw.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)

cv2.imshow("test", raw)
cv2.waitKey(0)


