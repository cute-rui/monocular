import numpy as np
import cv2

# Reference: https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

KNOWN_WIDTH = 10
KNOWN_HEIGHT = 8
FOCAL_LENGTH = 0


def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


# variable captured is a cv2.RotatedRect of target in raw frame
def proc_rotated_rect(captured_rotated_rect):
    inches = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, captured_rotated_rect[1][0])
    return "%.2fcm".format(inches * 30.48 / 12)


def proc_box(captured_box):
    left_point = min(captured_box, key=lambda point: point[0])
    right_point = max(captured_box, key=lambda point: point[0])
    width = abs(right_point[0] - left_point[0])

    inches = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, width)
    return "%.2fcm".format(inches * 30.48 / 12)


def proc_width(width):
    inches = distance_to_camera(KNOWN_WIDTH, FOCAL_LENGTH, width)
    return "%.2fcm".format(inches * 30.48 / 12)
