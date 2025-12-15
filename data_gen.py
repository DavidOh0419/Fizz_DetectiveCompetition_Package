import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from std_msgs.msg import String


def show_bgr(img, title=""):
    """Show BGR image with matplotlib."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def order_points(pts):
    """Order 4 points as TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

folder_path = os.path.join(BASE_DIR, "cap_samples")
save_path = os.path.join(BASE_DIR, "cropped_samples2")

def img_process(PATH, file_name):
    # 1. Read image
    img_full = cv2.imread(PATH)
    if img_full is None:
        raise RuntimeError(f"Could not read image at {PATH}")
    orig_full = img_full.copy()

    # ------------------------------------------------------------
    # STEP A: find BLUE sign and crop a ROI around it
    # ------------------------------------------------------------
    hsv_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)

    h, w = 0, 0
    adj = 0

    while (h*w < 5000):
        
        # your blue range
        lower_blue = (110, 220 - adj, 60 - adj)
        upper_blue = (130, 255 + adj, 150 + adj)

        adj = adj + 5

        blue_mask = cv2.inRange(hsv_full, lower_blue, upper_blue)

        kernel_blue = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_blue, iterations=2)

        #show_bgr(cv2.bitwise_and(img_full, img_full, mask=blue_mask), "Blue mask")

        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not blue_contours:
            raise RuntimeError("No blue contours found; adjust blue HSV thresholds.")

        # Largest blue contour = the blue frame
        blue_cnt = max(blue_contours, key=cv2.contourArea)

        # Bounding box of the blue frame → used only for cropping / later warp
        x, y, w, h = cv2.boundingRect(blue_cnt)
        blue_roi = orig_full[y:y + h, x:x + w]

    #show_bgr(blue_roi, "Cropped blue ROI")

    # ------------------------------------------------------------
    # NEW: build a filled mask for the entire blue region (frame + inside)
    # ------------------------------------------------------------
    h_full, w_full = hsv_full.shape[:2]
    blue_filled_mask = np.zeros((h_full, w_full), dtype=np.uint8)
    # Fill the blue contour area with 255 → everything inside the blue frame
    cv2.drawContours(blue_filled_mask, [blue_cnt], contourIdx=-1, color=255, thickness=-1)

    # Debug: see filled blue region
    # #show_bgr(cv2.bitwise_and(img_full, img_full, mask=blue_filled_mask), "Blue filled region")

    # ------------------------------------------------------------
    # STEP B (NEW): detect the inner gray/white board using edges
    #               INSIDE the blue ROI only
    # ------------------------------------------------------------

    roi = blue_roi.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection inside ROI
    edges = cv2.Canny(blur, 40, 130)

    # Remove edges touching the ROI border (removes blue frame)
    H_roi, W_roi = edges.shape
    border = 10
    edges[0:border, :] = 0
    edges[:, 0:border] = 0
    edges[H_roi-border:H_roi, :] = 0
    edges[:, W_roi-border:W_roi] = 0

    # Close gaps
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Debug: View detected edges
    # show_bgr(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "ROI edges")

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No contours found inside blue ROI (inner board not detected).")

    # Choose the largest internal contour
    inner_cnt = max(contours, key=cv2.contourArea)

    # Approximate corners
    peri = cv2.arcLength(inner_cnt, True)
    approx = cv2.approxPolyDP(inner_cnt, 0.02 * peri, True)

    # Fallback: rotated rectangle
    if len(approx) != 4:
        rect = cv2.minAreaRect(inner_cnt)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1,1,2).astype(np.int32)

    # 4 points (inside ROI coordinates)
    src = approx.reshape(4,2).astype("float32")

    # Visualize
    debug = roi.copy()
    for (cx,cy) in src.astype(int):
        cv2.circle(debug, (cx, cy), 8, (0,0,255), -1)
    #show_bgr(debug, "Detected inner board corners")

    # From this point on, we work *inside* the blue ROI as before
    img = blue_roi.copy()
    orig = img.copy()

    # ------------------------------------------------------------
    # STEP C: perspective transform using WHITE board corners
    # ------------------------------------------------------------

    src_ordered = order_points(src)
    (tl, tr, br, bl) = src_ordered

    width_top  = np.linalg.norm(tr - tl)
    width_bot  = np.linalg.norm(br - bl)
    max_width  = int(max(width_top, width_bot))

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height   = int(max(height_left, height_right))

    # Auto-sized destination rectangle
    dst_auto = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M_auto = cv2.getPerspectiveTransform(src_ordered, dst_auto)
    warped_auto = cv2.warpPerspective(orig, M_auto, (max_width, max_height))
    #show_bgr(warped_auto, "White board warped (auto size)")

    # If you want a fixed size for all boards:
    OUTPUT_WIDTH  = 1000
    OUTPUT_HEIGHT = 600

    dst_fixed = np.array([
        [0, 0],
        [OUTPUT_WIDTH - 1, 0],
        [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
        [0, OUTPUT_HEIGHT - 1]
    ], dtype="float32")

    M_fixed = cv2.getPerspectiveTransform(src_ordered, dst_fixed)
    warped = cv2.warpPerspective(orig, M_fixed, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    ##show_bgr(warped, "White board warped (fixed size)")

    h2, w2 = warped.shape[:2]
    warped = warped[:, 70:w2 - 70]

    name = os.path.splitext(os.path.basename(file_name))[0]
    cv2.imwrite( save_path + "/" + name + ".png", warped)

    
    
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    print(filename)
    img_process(file_path, filename)
        



