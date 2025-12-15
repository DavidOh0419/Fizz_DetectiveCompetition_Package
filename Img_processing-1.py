import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def show_bgr(img, title=""):
    """Show BGR image with matplotlib."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(5,5))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

PATH = "/home/fizzer/ros_ws/src/data_collection_pkg_team15_clueboard/cap_samples/1246000000000.jpg"

# 1. Read image
img = cv2.imread(PATH)
if img is None:
    raise RuntimeError(f"Could not read image at {PATH}")
orig_full = img.copy()

# ------------------------------------------------------------
# STEP A: Find & crop the biggest BLUE block (around R,G,B ≈ 2,1,103)
# ------------------------------------------------------------

# Convert to HSV
hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Tighter HSV range for DARK BLUE frame:
# - Hue:   100–140  (blue)
# - Sat:   80–120   (exclude low-S sky around 45)
# - Value: 40–200   (exclude very bright sky)
# H = 121
# S = 255
# V = 102

lower_blue = (110, 220, 60) #np.array([180, 80, 30], dtype=np.uint8)
upper_blue = (130, 255, 150) #np.array([255, 120, 50], dtype=np.uint8)

blue_mask = cv2.inRange(hsv_full, lower_blue, upper_blue)

# Morphological closing to fill small gaps
kernel_blue = np.ones((5, 5), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_blue, iterations=2)

# Optional debug: what we consider "blue"
show_bgr(cv2.bitwise_and(img, img, mask=blue_mask), "Blue areas mask")

# Find blue contours
blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(blue_contours) == 0:
    raise RuntimeError("No blue contours found; adjust your blue HSV thresholds.")

# Largest blue region = our blue block
blue_cnt = max(blue_contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(blue_cnt)

# Crop to the blue block
blue_roi = orig_full[y:y+h, x:x+w]
show_bgr(blue_roi, "Cropped biggest blue block")

# From this point on, we work *inside* the blue ROI
img = blue_roi.copy()
orig = img.copy()

# ------------------------------------------------------------
# STEP B: Within the blue block, find the WHITE area (clueboard)
# ------------------------------------------------------------

# 2. Threshold for white-ish clueboard (tweak numbers for your image)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Low saturation + relatively high value → white/grey area
lower_white = np.array([0, 0, 80], dtype=np.uint8)
upper_white = np.array([180, 80, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

# Close small gaps produced by text / texture
kernel = np.ones((7, 7), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Optional: see what we detected as white (inside blue block)
show_bgr(cv2.bitwise_and(img, img, mask=mask), "white area / clueboard mask (inside blue)")

# 3. Find contours of white
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    raise RuntimeError("No white contours found; adjust your white threshold.")

# Assume largest contour is the clueboard
cnt = max(contours, key=cv2.contourArea)

# 4. Approximate contour by polygon, aiming for 4 points (trapezoid)
peri = cv2.arcLength(cnt, True)
epsilon = 0.02 * peri
approx = cv2.approxPolyDP(cnt, epsilon, True)

# If we didn't get 4 points, relax epsilon until we do (simple heuristic)
eps_factor = 0.02
while len(approx) != 4 and eps_factor < 0.1:
    eps_factor += 0.01
    epsilon = eps_factor * peri
    approx = cv2.approxPolyDP(cnt, epsilon, True)

if len(approx) != 4:
    raise RuntimeError(f"Could not get 4 corners, got {len(approx)}. Try tweaking thresholds/epsilon.")

# approx has shape (4,1,2) → reshape to (4,2)
src = approx.reshape(4, 2).astype("float32")

# Visualize the detected trapezoid (inside the blue crop)
debug = orig.copy()
for (cx, cy) in src.astype(int):
    cv2.circle(debug, (cx, cy), 8, (0, 0, 255), -1)
show_bgr(debug, "Detected 4 corners (trapezoid) inside blue block")

# 5. Order points: TL, TR, BR, BL
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

src_ordered = order_points(src)

# 6. Compute output rectangle size from the trapezoid itself
(tl, tr, br, bl) = src_ordered

width_top  = np.linalg.norm(tr - tl)
width_bot  = np.linalg.norm(br - bl)
max_width  = int(max(width_top, width_bot))

height_left  = np.linalg.norm(bl - tl)
height_right = np.linalg.norm(br - tr)
max_height   = int(max(height_left, height_right))

# 7. Define destination (auto-sized rectangle)
dst_auto = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]
], dtype="float32")

# 8. Perspective transform: trapezoid → rectangle (inside blue ROI)
M_auto = cv2.getPerspectiveTransform(src_ordered, dst_auto)
warped_auto = cv2.warpPerspective(orig, M_auto, (max_width, max_height))

show_bgr(warped_auto, "Perspective-corrected clueboard (auto size)")

# --- FIXED OUTPUT SIZE (same for all warped images) ---
OUTPUT_WIDTH  = 1000
OUTPUT_HEIGHT = 600

dst = np.array([
    [0, 0],
    [OUTPUT_WIDTH - 1, 0],
    [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
    [0, OUTPUT_HEIGHT - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(src_ordered, dst)
warped = cv2.warpPerspective(orig, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

# Crop side edges
h2, w2 = warped.shape[:2]
warped = warped[:, 50:w2-50]

show_bgr(warped, "Warped (Fixed Size) + side-edges cropped")

# Divide upper and lower halves of the clue board
h, w = warped.shape[:2]
mid = h // 2

warped_upper = warped[0:mid, :]
warped_lower = warped[mid:h, :]

show_bgr(warped_lower, "Lower")
show_bgr(warped_upper, "Upper")

warped_upper = warped_upper[:, w//2 - 80:w//2]

show_bgr(warped_upper, "cropped")
