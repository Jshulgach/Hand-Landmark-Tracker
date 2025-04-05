####  Copyright 2025 Jonathan Shulgach jonathan@shulgach.com
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
import time
import cv2 as cv
import argparse
import numpy as np
import mediapipe as mp
from inmoov import InMoovArm

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
hand_lm = mp_hands.HandLandmark

def detect_landmarks(frame, img_size, hands, pose, visualize=True):
    """ Perform hand and body landmark detection using the passed-in detector models"""
    # Resize the frame
    frame = cv.resize(frame, (img_size[0], img_size[1]))
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # Convert the frame to RGB
    image.flags.writeable = False
    #body_results = pose.process(image) # Perform pose detection
    body_results = None
    hand_results = hands.process(image) # Hand detection
    image.flags.writeable = True
    # Convert back to BGR for visualization
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    if visualize:
        # Draw the hand landmarks
        if hand_results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Only use the specified hand type
                if "Right" == hand_results.multi_handedness[i].classification[0].label:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 205, 0), thickness=2, circle_radius=4),
                    )

        # Draw the body pose landmarks
        #mp_drawing.draw_landmarks(image, body_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        #)

    return image, body_results, hand_results

def compute_triangle(image, point1, point2, point3, visualize=False):
    """
    Draw a triangle on the image given three points.
    """
    # Convert points to pixel coordinates
    p1 = (int(point1.x * image.shape[1]), int(point1.y * image.shape[0]))
    p2 = (int(point2.x * image.shape[1]), int(point2.y * image.shape[0]))
    p3 = (int(point3.x * image.shape[1]), int(point3.y * image.shape[0]))

    # Compute the angles given points
    if p1 == p2 or p2 == p3 or p3 == p1:
        # If any two points are the same, return None (invalid triangle)
        print("Invalid triangle: Two or more points are the same.")
        return None

    if visualize:
        # Draw the triangle
        cv.line(image, p1, p2, (255, 0, 0), 2)
        cv.line(image, p2, p3, (255, 255, 0), 2)
        cv.line(image, p3, p1, (255, 0, 255), 2)

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    angle_a = np.arccos(
        (np.linalg.norm(p2 - p3) ** 2 + np.linalg.norm(p1 - p3) ** 2 - np.linalg.norm(p1 - p2) ** 2) /
        (2 * np.linalg.norm(p2 - p3) * np.linalg.norm(p1 - p3))
    ) * (180 / np.pi)
    angle_b = np.arccos(
        (np.linalg.norm(p1 - p2) ** 2 + np.linalg.norm(p2 - p3) ** 2 - np.linalg.norm(p1 - p3) ** 2) /
        (2 * np.linalg.norm(p1 - p2) * np.linalg.norm(p2 - p3))
    ) * (180 / np.pi)
    angle_c = 180 - angle_a - angle_b

    return [angle_a, angle_b, angle_c]

def compute_finger_angles(hand_results, handtype="Right"):
    """
    Gets the triangles between all finger, and computes the angles for each finger's triangle
    """
    if hand_results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
            # Only use the specified hand type
            if handtype != hand_results.multi_handedness[i].classification[0].label:
                continue

            # Get the fingertip, knuckle, and palm landmarks
            finger_tips = {
                'thumb': hand_landmarks.landmark[hand_lm['THUMB_TIP']],          # Thumb Tip
                'index': hand_landmarks.landmark[hand_lm['INDEX_FINGER_TIP']],   # Index finger Tip
                'middle': hand_landmarks.landmark[hand_lm['MIDDLE_FINGER_TIP']], # Middle finger Tip
                'ring': hand_landmarks.landmark[hand_lm['RING_FINGER_TIP']],     # Ring finger Tip
                'pinky': hand_landmarks.landmark[hand_lm['PINKY_TIP']],          # Pinky finger Tip
            }
            finger_mpc = {
                'thumb': hand_landmarks.landmark[hand_lm['THUMB_MCP']],          # Thumb metacarpal (base of the thumb)
                'index': hand_landmarks.landmark[hand_lm['INDEX_FINGER_MCP']],   # Index finger metacarpal (base of the index finger)
                'middle': hand_landmarks.landmark[hand_lm['MIDDLE_FINGER_MCP']], # Middle finger metacarpal (base of the middle finger)
                'rind': hand_landmarks.landmark[hand_lm['RING_FINGER_MCP']],     # Ring finger metacarpal (base of the ring finger)
                'pinky': hand_landmarks.landmark[hand_lm['PINKY_MCP']],          # Pinky metacarpal (base of the pinky finger)
            }
            palm = [hand_landmarks.landmark[hand_lm['WRIST']],
                             hand_landmarks.landmark[hand_lm['INDEX_FINGER_MCP']],
                             hand_landmarks.landmark[hand_lm['PINKY_MCP']]
            ]

            # Compute the triangles between the fingers and palm
            thumb_angles = compute_triangle(image, finger_tips['thumb'], finger_mpc['thumb'], palm[0])  # Thumb triangle
            index_angles = compute_triangle(image, finger_tips['index'], finger_mpc['index'], palm[0])  # Index finger triangle
            middle_angles = compute_triangle(image, finger_tips['middle'], finger_mpc['middle'], palm[0])  # Middle finger triangle
            ring_angles = compute_triangle(image, finger_tips['ring'], finger_mpc['rind'], palm[0])  # Ring finger triangle
            pinky_angles = compute_triangle(image, finger_tips['pinky'], finger_mpc['pinky'], palm[0])  # Pinky finger triangle

            # Angle b for each triangle correlates with a joint angle command
            finger_angles = [thumb_angles[1], index_angles[1], middle_angles[1], ring_angles[1], pinky_angles[1]]

            # Draw a blue dot on the palm landmarks
            if palm is not None:
                for point in palm:
                    x, y = int(point.x * image.shape[1]), int(point.y * image.shape[0])
                    cv.circle(image, (x, y), 4, (255, 0, 0), -1)

            return finger_angles

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Human Pose Estimation and Hand Landmark Tracking on a video file or webcam stream')
    parser.add_argument('--camera_id', type=str, default='0', help='Camera ID or video file path')
    parser.add_argument('--img_size', type=tuple, default=(1080, 720), help='Image size to resize the input frame (width, height)')
    parser.add_argument('--detection_confidence', type=float, default=0.5, help='Minimum confidence for hand detection (0.0 to 1.0)')
    parser.add_argument('--tracking_confidence', type=float, default=0.5, help='Minimum confidence for hand tracking (0.0 to 1.0)')
    parser.add_argument('--port', type=str, default='COM3', help='Serial port to connect to the InMoov arm controller (if using serial)')
    parser.add_argument('--baudrate', type=int, default=115200, help='Baudrate for serial communication with the InMoov arm controller')
    args = parser.parse_args()

    # Initialize the InMoov arm controller
    arm = InMoovArm(port=args.port, baudrate=args.baudrate)

    # Open the video capture
    camera_id = int(args.camera_id) if args.camera_id.isdigit() else args.camera_id
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Create detectors
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=args.detection_confidence, min_tracking_confidence=args.tracking_confidence)
    pose = mp_pose.Pose(min_detection_confidence=args.detection_confidence, min_tracking_confidence=args.tracking_confidence)

    print("Running hand and arm tracking. Press 'q' to quit.")
    start_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally for mediapipe, and perform landmark detection
        frame = cv.flip(frame, 1)
        image, _, hand_results = detect_landmarks(frame, args.img_size, hands, pose)

        # If 100ms have passed, send the joint values to the InMoov arm controller
        if time.time() - start_t > 0.1:
            start_t = time.time()
            # Get the finger joint angles
            finger_angles = compute_finger_angles(hand_results)

            # Flip the angular value to match the servo's range (0-180 degrees) for the InMoov arm
            if finger_angles is not None:
                finger_angles = [(180 - i) for i in finger_angles]

            # Get wrist joint
            wrist = 90.0

            # Send the joint values to the 3d-printed arm for direct-servo control
            if finger_angles is not None:
                msg = 'set_joints:' + str(int(wrist)) + ':' +  ':'.join([str(int(angle)) for angle in finger_angles]) # Format the message
                print(f"Sending: {msg}")
                arm.send_message(msg)

        # Display the image
        cv.imshow('Hand and Arm Tracker', image)

        # Exit on 'q' key press
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()
    print("Exiting the Hand and Arm Tracker application...")
