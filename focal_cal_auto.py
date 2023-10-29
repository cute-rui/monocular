import cv2

# **inaccurate!** deviation will be magnified when getting edge box

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    c = max(cnts, key=cv2.contourArea)

    return cv2.minAreaRect(c)

def get_focal_img(KNOWN_DISTANCE, KNOWN_WIDTH, IMGPATH):
    image = cv2.imread(IMGPATH)
    marker = find_marker(image)
    return (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

def get_focal_raw(KNOWN_DISTANCE, KNOWN_WIDTH, IMG):
    marker = find_marker(IMG)
    return (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


