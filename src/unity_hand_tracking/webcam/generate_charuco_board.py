"""
generate_charuco_board.py — Generate a printable ChArUco board image.

Run this script to create a high-resolution PNG of the ChArUco board
that matches the calibration settings in config.py. Print it on paper,
mount it on a flat surface (cardboard, clipboard, etc.), and use it
for camera calibration.

Usage:
    python generate_charuco_board.py
"""

import os

import cv2

try:
    from .config import (
        ARUCO_DICT,
        CALIBRATION_DIR,
        CHARUCO_MARKER_LENGTH,
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
    )
except ImportError:
    from config import (
        ARUCO_DICT,
        CALIBRATION_DIR,
        CHARUCO_MARKER_LENGTH,
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_SQUARES_X,
        CHARUCO_SQUARES_Y,
    )


def main():
    dict_id = getattr(cv2.aruco, ARUCO_DICT, cv2.aruco.DICT_5X5_250)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH,
        CHARUCO_MARKER_LENGTH,
        aruco_dict,
    )

    # Generate a high-resolution image (300 DPI at ~A4 size)
    # Each square will be ~200 pixels for crisp printing
    pixels_per_square = 200
    img_width = CHARUCO_SQUARES_X * pixels_per_square
    img_height = CHARUCO_SQUARES_Y * pixels_per_square
    margin = 40  # White border for printing

    board_img = board.generateImage(
        (img_width, img_height),
        marginSize=margin,
    )

    # Save to calibration directory
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    output_path = os.path.join(CALIBRATION_DIR, "charuco_board.png")
    cv2.imwrite(output_path, board_img)

    print(f"ChArUco board saved to: {output_path}")
    print(f"  Board: {CHARUCO_SQUARES_X} x {CHARUCO_SQUARES_Y} squares")
    print(f"  Square size: {CHARUCO_SQUARE_LENGTH * 1000:.0f} mm")
    print(f"  Marker size: {CHARUCO_MARKER_LENGTH * 1000:.0f} mm")
    print(f"  Dictionary: {ARUCO_DICT}")
    print(f"  Image: {img_width + 2 * margin} x {img_height + 2 * margin} px")
    print()
    print(
        "Print this image at 100% scale (no fit-to-page) and mount on a flat surface."
    )
    print(
        "Measure the printed square size and update CHARUCO_SQUARE_LENGTH in config.py if needed."
    )

    # Also show it
    display = cv2.resize(board_img, (img_width // 2, img_height // 2))
    cv2.imshow("ChArUco Board - Press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
