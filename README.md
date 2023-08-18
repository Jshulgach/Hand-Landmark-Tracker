# Hand Landmark Tracker &nbsp;[![](https://img.shields.io/badge/python-3.8.5-blue.svg)](https://www.python.org/downloads/)
Hand Landmark Tracker makes human-computer interaction (HCI) simpler with the use of hand commands. The hand is the most versatile controller on can use.
This code uses the amazing features of Google's machine learning suite [MediaPipe](https://developers.google.com/mediapipe), a media-based ML package for classification and recognition with neural networks.
The project was also inspired by xenon-19's [Gesture Controlled Virtual Mouse](https://github.com/xenon-19/Gesture-Controlled-Virtual-Mouse) project.

This program uses openCV and mediapipe to acquire hand landmarks and post/gesture tracking commands to stream to a [Robot Web Server](). 

Detector in action.
<figure>
  <img src="https://github.com/Jshulgach/Hand-Landmark-Tracker/blob/main/media/hands.gif" alt="Hand" width="711" height="400"><br>
  <figcaption>Landmark tracking. Multi-hand classification and landmark identification.</figcaption>
</figure>


## Getting Started
Assuming you have python>=3.8 installed on your PC and the PATH environment variable set up, continue with installation

### Install python package dependencies:
~~~
pip install python-opencv mediapipe python-socket
~~~

### Clone repository
~~~
git clone https://github.com/Jshulgach/Hand-Landmark-Tracker.git
cd Hand-Landmark-Tracker
~~~

### Run main script
~~~
python main.py
~~~

