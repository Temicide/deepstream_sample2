from typing import Optional

import cv2
import numpy as np


# OpenCV HSV ranges: H=0-179, S=0-255, V=0-255.
COLOR_BLACK_V = 85
COLOR_WHITE_V = 160
COLOR_NEUTRAL_S = 50
ENABLE_TWO_TONE = False


def detect_color(vehicle_bgr_img: np.ndarray) -> str:
    """Return a rule-based vehicle color label from a BGR crop."""
    if vehicle_bgr_img is None or vehicle_bgr_img.size == 0:
        return "Unknown"

    h, w = vehicle_bgr_img.shape[:2]
    roi = vehicle_bgr_img[int(h * 0.10) : int(h * 0.70), int(w * 0.15) : int(w * 0.85)]
    if roi.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    valid_mask = (v_channel > 40) & ~((v_channel > 220) & (s_channel < 30))

    if ENABLE_TWO_TONE:
        two_tone = detect_two_tone(h_channel, s_channel, v_channel, valid_mask)
        if two_tone:
            return two_tone

    if np.sum(valid_mask) < 20:
        if np.mean(v_channel) < 50:
            return "Black"
        return "Unknown"

    return calculate_color_votes(h_channel[valid_mask], s_channel[valid_mask], v_channel[valid_mask])


def detect_two_tone(
    h_channel: np.ndarray,
    s_channel: np.ndarray,
    v_channel: np.ndarray,
    valid_mask: np.ndarray,
) -> Optional[str]:
    edges = cv2.Canny(v_channel, 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=valid_mask.astype(np.uint8) * 255)

    roi_h, roi_w = v_channel.shape[:2]
    min_length = int(max(roi_h, roi_w) * 0.4)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_length, maxLineGap=20)
    if lines is None:
        return None

    lines = sorted(lines, key=lambda line: np.hypot(line[0][2] - line[0][0], line[0][3] - line[0][1]), reverse=True)
    y_grid, x_grid = np.indices((roi_h, roi_w))
    cx, cy = roi_w / 2, roi_h / 2

    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_len = np.hypot(x2 - x1, y2 - y1)
        if line_len < 1e-5:
            continue

        dist_to_center = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / line_len
        if dist_to_center > min(cx, cy) * 0.8:
            continue

        side_mask = ((y2 - y1) * x_grid - (x2 - x1) * y_grid + x2 * y1 - y2 * x1) > 0
        mask1 = valid_mask & side_mask
        mask2 = valid_mask & ~side_mask

        if np.sum(mask1) <= 100 or np.sum(mask2) <= 100:
            continue

        color1 = calculate_color_votes(h_channel[mask1], s_channel[mask1], v_channel[mask1])
        color2 = calculate_color_votes(h_channel[mask2], s_channel[mask2], v_channel[mask2])
        if color1 == color2 or color1 == "Unknown" or color2 == "Unknown":
            continue

        neutrals = {"White", "Silver/Gray", "Black"}
        if color1 in neutrals and color2 in neutrals and "Black" not in {color1, color2}:
            continue

        colors = sorted([color1, color2])
        return f"{colors[0]}+{colors[1]}"

    return None


def calculate_color_votes(valid_h: np.ndarray, valid_s: np.ndarray, valid_v: np.ndarray) -> str:
    if len(valid_v) == 0:
        return "Unknown"

    votes = {
        "Black": int(np.sum(valid_v < COLOR_BLACK_V)),
        "White": int(np.sum((valid_v > COLOR_WHITE_V) & (valid_s < COLOR_NEUTRAL_S))),
    }

    remain_mask = (valid_v >= COLOR_BLACK_V) & ~((valid_v > COLOR_WHITE_V) & (valid_s < COLOR_NEUTRAL_S))
    remain_s = valid_s[remain_mask]
    remain_h = valid_h[remain_mask]

    votes["Silver/Gray"] = int(np.sum(remain_s < COLOR_NEUTRAL_S))

    chroma_mask = remain_s >= COLOR_NEUTRAL_S
    color_h = remain_h[chroma_mask]

    if len(color_h) > 0:
        votes["Red"] = int(np.sum((color_h < 10) | (color_h > 165)))
        votes["Orange"] = int(np.sum((color_h >= 10) & (color_h < 25)))
        votes["Yellow"] = int(np.sum((color_h >= 25) & (color_h < 38)))
        votes["Green"] = int(np.sum((color_h >= 38) & (color_h < 85)))
        votes["Blue"] = int(np.sum((color_h >= 85) & (color_h < 130)))
        votes["Purple"] = int(np.sum((color_h >= 130) & (color_h <= 165)))

    return max(votes, key=votes.get)
