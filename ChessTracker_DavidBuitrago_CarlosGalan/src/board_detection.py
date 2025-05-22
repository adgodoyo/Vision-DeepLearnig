import cv2
import numpy as np


def detect_board_corners(frame):
    """
    Detect the outer corners of the chessboard using contour detection.
    Returns found (bool), corners (np.array) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4:
            corners = np.array([pt[0] for pt in approx], dtype='float32')
            ordered = order_corners(corners)
            return True, ordered

    return False, None


def order_corners(pts):
    """
    Orders 4 points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left

    return rect


def warp_board(frame, corners, square_size=50):
    """
    Applies a perspective transform to get a top-down view of the board.
    """
    dst_pts = np.array([
        [0, 0],
        [square_size * 8, 0],
        [square_size * 8, square_size * 8],
        [0, square_size * 8]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    top_down = cv2.warpPerspective(frame, M, (square_size * 8, square_size * 8))

    return top_down

def draw_corners(frame, corners):
    """
    Draws the detected corners on the frame for visualization.
    """
    for pt in corners:
        cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
    return frame