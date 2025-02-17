import cv2
import time
import serial
import threading
import numpy as np

from utils.tracking import HandTracker
from utils.miniarm import MiniArmClient

class VirtualSpacemouse(object):
    """ A class that serves as a velocity controller for a spacemouse using hand tracking as the control mode

    Parameters
    ----------
    name (str) : Name of the virtual spacemouse
    port (str) : Serial port to connect to
    baudrate (int) : Baudrate for the serial connection
    visualize (bool) : Enable/disable visualization
    command_rate (int) : Command rate for the spacemouse
    verbose (bool) : Enable/disable verbose output
    """
    def __init__(self, name='VirtualSpacemouse', port='COM3', baudrate=9600, visualize=False, command_rate=10, verbose=False):
        self.name = name
        self.visualize = visualize
        self.command_rate = command_rate
        self.verbose = verbose

        # Initialize some variables
        self.all_stop = False
        self.palm_pos = []
        self.ref_pos = None
        self.ref_set = False
        self.pos_limit = 10
        self.std_lim = 5
        self.last_command_time = time.time()
        self.last_gripper_cmd = 0
        self.robot_ref_pose = [0,0,0,0,0,0]

        # Initialize hand tracker
        self.tracker = HandTracker(max_hands=1, confidence=0.8, visualize=False, verbose=self.verbose)

        # Setup robot connection using the MiniArmClient class, which should make it easier to get robot data
        self.robot = MiniArmClient(port=port, baudrate=baudrate)
        self.logger("Virtual Spacemouse initialized! Press Ctrl+C to stop...\n")

    def logger(self, *argv, warning=False):
        """ Robust logging function to track events"""
        msg = ''.join(argv)
        if warning: msg = '(Warning) ' + msg
        print("[{:.3f}][{}] {}".format(time.monotonic(), self.name, msg))

    def serial_listen(self):
        """ Listens for incoming serial data. Useful for debugging, disable otherwise """
        while not self.all_stop:
            if self.robot.s.in_waiting > 0:
                print(self.robot.s.readline().decode().strip())

            time.sleep(0.01)

    def start(self):
        """ Start main threads. Note that the serial thread is useful for debugging, but it needs to be disabled if we
        want to receive and use data from the connected robot """
        #serial_thread = threading.Thread(target=self.serial_listen, daemon=True)
        #serial_thread.start()

        update_thread = threading.Thread(target=self.update, daemon=True)
        update_thread.start()

        # Prevent the main thread from exiting
        while not self.all_stop:
            time.sleep(0.1)

    def update(self):
        """ Process hand tracking and update CV window """
        while True:
            # Get the image and perform hand detection
            image = self.tracker.get_image(flip_image=True)
            image = self.tracker.detect_hands(image)
            hand_landmarks = self.tracker.hand_landmarks

            if hand_landmarks:
                for hand_landmark in hand_landmarks:
                    thumb_tip = hand_landmark.landmark[4] # Thumb tip
                    index_tip = hand_landmark.landmark[8] # Index finger tip
                    palm = hand_landmark.landmark[0]  # Palm

                    # Save the palm position
                    x = int(palm.x * image.shape[1])
                    y = int(palm.y * image.shape[0])
                    self.palm_pos.append((x, y))
                    if len(self.palm_pos) > self.pos_limit:
                        self.palm_pos.pop(0)

                    # If we have enough palm positions, compute the average and std deviation
                    if len(self.palm_pos) == self.pos_limit:
                        avg_pos = np.mean(self.palm_pos, axis=0)
                        std_pos = np.std(self.palm_pos, axis=0)

                        # The Reference point gets set if the std is less than 10
                        if not np.all(std_pos < self.std_lim) and not self.ref_set:
                            self.ref_pos = avg_pos
                        else:
                            # Get the robot current pose, only once
                            if self.robot_ref_pose is None:
                                self.robot_ref_pose = list(self.robot.get_current_pose())
                                print(self.robot_ref_pose)
                            self.ref_set = True

                        if self.ref_pos is not None:
                            # Get the normalized pixeldifference between the average pose of the palm position and the reference pose
                            diff = (avg_pos - self.ref_pos) / 100
                            diff = np.clip(diff, -1, 1) * -1  # Don't need large values

                        # Compute distance between thumb and index finger
                        thumb_index_gap_px = np.linalg.norm(
                            np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
                        )

                        # Add circle to the palm position
                        color = (0, 255, 0) if self.ref_set else (0, 0, 255)  # Green if set, Red if not
                        cv2.circle(image, (x, y), 8, color, -1)

                        # Draw a red arrow, the vector is the distance between the new palm position and the reference
                        if self.ref_set and self.ref_pos is not None and isinstance(self.ref_pos, np.ndarray) and len(self.ref_pos) == 2:
                            cv2.arrowedLine(image, tuple(map(int, self.ref_pos)), tuple(map(int, avg_pos)), (0, 0, 255), 2)


                        # Draw a circle at the thumb and index tip and a line between them
                        cv2.circle(image, (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])), 8, (255, 0, 0), -1)
                        cv2.circle(image, (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])), 8, (255, 0, 0), -1)
                        cv2.line(image, (int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])),
                                 (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])), (255, 0, 0), 2)

                        if self.verbose:
                            cv2.putText(image, f'Std. Deviation: {std_pos}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(image, f'Palm Position: {avg_pos}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(image, f'Reference: {self.ref_pos}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(image, f'Diff: {diff}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            cv2.putText(image, f'Thumb-Index Gap: {thumb_index_gap_px:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        if self.ref_set:
                            current_time = time.time()
                            if current_time - self.last_command_time > 1.0 / self.command_rate:
                                self.last_command_time = current_time

                                # Old version, sending a delta command like a velocity controller
                                # diff = diff / 100 # diff of 1 == 10mm step size for delta commands
                                # msg = f'set_delta_pose:[0.0,{diff[0]:.3f},{diff[1]:.3f}];'
                                #print(msg)
                                #self.robot.send_message(msg)

                                # Send a pose command
                                x_cmd = self.robot_ref_pose[0]
                                y_cmd = self.robot_ref_pose[1] + 0.05 * diff[0]  # 50mm for ~1
                                z_cmd = self.robot_ref_pose[2] + 0.05 * diff[1]
                                msg = f'set_pose:[{x_cmd:.3f}, {y_cmd:.3f}, {z_cmd:.3f}];'
                                print(msg)
                                self.robot.send_message(msg)

                                # Send gripper command, correlate with thumb-index gap as 0.02-0.22px diff for 0-130 degrees
                                gripper_cmd = int(0 + (thumb_index_gap_px - 0.02) / (0.22 - 0.02) * (130 - 0))
                                if gripper_cmd != self.last_gripper_cmd:
                                    self.last_gripper_cmd = gripper_cmd
                                    msg = f'set_gripper:{gripper_cmd};'
                                    print(msg)
                                    self.robot.send_message(msg)


            else:
                # Reset the reference pose if the hand is out of view
                self.ref_pos = None
                self.ref_set = False
                self.robot_ref_pose = None
                self.palm_pos.clear()

            if self.visualize:
                cv2.imshow('Virtual Spacemouse', image)
                key = cv2.waitKey(1)
                if key & 0xFF == 27: # Esc key
                    self.all_stop = True
                    break

        self.tracker.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        """ Sets a flag to stop running all tasks """
        self.all_stop = True
        self.robot.disconnect()
        self.tracker.cap.release()
        cv2.destroyAllWindows()
