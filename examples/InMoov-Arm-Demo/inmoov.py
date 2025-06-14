import time
import serial
import threading
import numpy as np

__version__ = '0.1.0'
__author__ = 'Jonathan Shulgach'

class InMoovArm(object):
    """ A class that serves as a client for the InMoov arm and acts like a driver

    Parameters
    ----------
    name (str) : Name of the client
    port (str) : Serial port to connect to
    baudrate (int) : Baudrate for the serial connection
    visualize (bool) : Enable/disable visualization
    verbose (bool) : Enable/disable verbose output
    """

    def __init__(self, name='InMoovClient', port='COM3', baudrate=9600, command_delimiter=';', verbose=False):
        self.name = name
        self.command_delimiter = command_delimiter
        self.verbose = verbose

        # Setup serial communication
        self.s = serial.Serial(port, baudrate, timeout=1)
        self.connected = True if self.s.is_open else False

        # Print out any bytes currently in the buffer
        _ = self.get_buffer()

        self.home() # Set to home position upon successful connection

        self.logger(f"{self.name} initialized\n")

    def logger(self, *argv, warning=False):
        """ Robust logging function to track events"""
        msg = ''.join(argv)
        if warning: msg = '(Warning) ' + msg
        print("[{:.3f}][{}] {}".format(time.monotonic(), self.name, msg))

    def send_message(self, message):
        """ Sends a string message to the Pico over serial
        """
        if self.s and self.s.is_open:
            if not message.endswith(self.command_delimiter):
                message += self.command_delimiter  # Add terminator character to the message
            try:
                self.s.write(message.encode())
                if self.verbose:
                    print(f"Sent message to Pico: {message}")
                time.sleep(0.01) # Mandatory 10ms delay to prevent buffer overflow
            except serial.SerialException as e:
                print(f"Error sending message: {e}")
        else:
            print("Serial connection not available or not open.")

    def get_buffer(self):
        """Read all available bytes from the serial connection and return them as a string if any
        """
        if self.s and self.s.is_open:
            try:
                if self.s.in_waiting > 0:
                    msg = ""
                    while self.s.in_waiting > 0:
                        msg += self.s.readline().decode().strip()
                        time.sleep(0.01) # Mandatory 10ms delay to prevent buffer overflow
                    return msg
            except serial.SerialException as e:
                print(f"Error reading message: {e}")

    def get_joints(self):
        """ Returns the current joints of the InMoov arm
        """
        # TO-DO
        return None

    def home(self):
        """ Sends the robot arm to the home position"""
        self.send_message("home;")

    def send_ctrl_c(self):
        """ Sends the Ctrl+C command to the robot arm"""
        self.s.write(b'\x03')
        time.sleep(0.01)

    def send_ctrl_d(self):
        """ Sends the Ctrl+D command to the robot arm"""
        self.s.write(b'\x04')
        time.sleep(0.01)

    def disconnect(self):
        """ A function that stops the client and closes the serial connection"""
        self.s.close()
        self.logger("Serial connection closed...")

    def set_debug_mode(self, mode):
        """ Set the debug mode of the robot arm

        Parameters:
        -----------
        mode    (bool, str): The debug mode to set (true/false or 'on'/'off'
        """
        self.send_message(f"debug:{mode}")
