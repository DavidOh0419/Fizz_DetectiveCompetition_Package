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


PATH = "/home/fizzer/ros_ws/src/data_collection_pkg_team15/cap_samples/B_DXCIATVVRRIG.jpg"

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

    show_bgr(cv2.bitwise_and(img_full, img_full, mask=blue_mask), "Blue mask")

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not blue_contours:
        raise RuntimeError("No blue contours found; adjust blue HSV thresholds.")

    # Largest blue contour = the blue frame
    blue_cnt = max(blue_contours, key=cv2.contourArea)

    # Bounding box of the blue frame → used only for cropping / later warp
    x, y, w, h = cv2.boundingRect(blue_cnt)
    blue_roi = orig_full[y:y + h, x:x + w]
    print(adj)

show_bgr(blue_roi, "Cropped blue ROI")

# ------------------------------------------------------------
# NEW: build a filled mask for the entire blue region (frame + inside)
# ------------------------------------------------------------
h_full, w_full = hsv_full.shape[:2]
blue_filled_mask = np.zeros((h_full, w_full), dtype=np.uint8)
# Fill the blue contour area with 255 → everything inside the blue frame
cv2.drawContours(blue_filled_mask, [blue_cnt], contourIdx=-1, color=255, thickness=-1)

# Debug: see filled blue region
# show_bgr(cv2.bitwise_and(img_full, img_full, mask=blue_filled_mask), "Blue filled region")

# ------------------------------------------------------------
# STEP B: find the WHITE board *only inside* the blue region
# ------------------------------------------------------------

# Threshold for white on the FULL image
lower_white = (0, 0, 80)      # tweak as needed
upper_white = (179, 80, 255)   # low saturation, high value

white_mask_full = cv2.inRange(hsv_full, lower_white, upper_white)

# Keep only white pixels that are INSIDE the blue contour
white_in_blue = cv2.bitwise_and(white_mask_full, white_mask_full, mask=blue_filled_mask)

# Clean up
kernel = np.ones((3, 3), np.uint8)
white_in_blue = cv2.morphologyEx(white_in_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
white_in_blue = cv2.morphologyEx(white_in_blue, cv2.MORPH_OPEN, kernel, iterations=1)

# Debug: this is the white board strictly inside blue frame
show_bgr(cv2.bitwise_and(img_full, img_full, mask=white_in_blue), "White inside blue")

contours, _ = cv2.findContours(white_in_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

H_mask, W_mask = white_in_blue.shape
candidates = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:      # ignore tiny noise
        continue

    x2, y2, w2, h2 = cv2.boundingRect(cnt)

    # Reject contours that hug the FULL-image border – they can't be fully inside blue
    border_margin = 3
    if (x2 <= border_margin or y2 <= border_margin or
        x2 + w2 >= W_mask - border_margin or
        y2 + h2 >= H_mask - border_margin):
        continue

    candidates.append(cnt)

# If we rejected everything, fall back to the largest
if candidates:
    white_cnt_full = max(candidates, key=cv2.contourArea)
else:
    white_cnt_full = max(contours, key=cv2.contourArea)

# Approximate the contour as a 4-point polygon; if that fails, use minAreaRect
peri = cv2.arcLength(white_cnt_full, True)
approx = cv2.approxPolyDP(white_cnt_full, 0.02 * peri, True)

if len(approx) != 4:
    rect = cv2.minAreaRect(white_cnt_full)
    box = cv2.boxPoints(rect)
    approx = box.reshape(-1, 1, 2).astype(np.int32)

# approx is in FULL-IMAGE coordinates
src_full = approx.reshape(4, 2).astype("float32")

# Convert these points into coordinates relative to the blue ROI
# (so the rest of your code can still work on blue_roi as before)
src = src_full - np.array([x, y], dtype="float32")

# Visualize detected white-board corners inside blue ROI
debug_roi = blue_roi.copy()
for (cx, cy) in src.astype(int):
    cv2.circle(debug_roi, (cx, cy), 6, (0, 0, 255), -1)
show_bgr(debug_roi, "Detected 4 corners of WHITE board (inside blue frame)")

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
show_bgr(warped_auto, "White board warped (auto size)")

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
show_bgr(warped, "White board warped (fixed size)")

# ------------------------------------------------------------
# STEP D: crop side edges and split upper / lower halves
# (your existing code continues from here unchanged)
# ------------------------------------------------------------

h2, w2 = warped.shape[:2]
warped = warped[:, 50:w2 - 50]

h, w = warped.shape[:2]
mid = h // 2

warped_upper = warped[0:mid, :]
warped_lower = warped[mid:h, :]

show_bgr(warped_lower, "Lower")
show_bgr(warped_upper, "Upper")

warped_upper_center = warped_upper[:, w // 2 - 80:w // 2]
show_bgr(warped_upper_center, "Upper center cropped")

section_width = w // 12
sections = []
for i in range(12):
    x0 = i * section_width
    x1 = (i + 1) * section_width
    section_img = warped_lower[:, x0:x1]
    sections.append(section_img)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "license_plate_char_model.keras")
model = load_model(MODEL_PATH)

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_, H_in, W_in, C = model.input_shape

def preprocess_char_image(img_in):
    if img_in.ndim == 3:
        img = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    else:
        img = img_in

    img = cv2.resize(img, (W_in, H_in))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def predict_character(img_in):
    x = preprocess_char_image(img_in)
    probs = model.predict(x)[0]
    idx = int(np.argmax(probs))
    char = CLASSES[idx]
    conf = float(probs[idx])
    return char, conf

clue = ""
for i in range(12):
    pred_char, conf = predict_character(sections[i])
    if pred_char == "0":
        pred_char = " "
    clue = clue + pred_char

print(clue)
