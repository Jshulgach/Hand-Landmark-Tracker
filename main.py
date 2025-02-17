"""
Main entry point for the hand tracking application. It allows the user to choose between CLI and GUI modes.
"""
import sys
import argparse
from utils.tracking import HandTracker
from utils.virtualspacemouse import VirtualSpacemouse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hand Tracking Application')
    parser.add_argument('--mode', type=str, choices=['gui', 'cli', 'spacemouse'], default='spacemouse', help='Choose whether to run the GUI or CLI mode')
    parser.add_argument('--port', type=str, default='COM5', help='Serial port to connect to')
    parser.add_argument('--verbose', type=bool, default=False, help='Enable verbose output')
    args = parser.parse_args()

    try:

        if args.mode == 'cli':
            print("Running in CLI mode...")
            tracker = HandTracker(visualize=args.visualize, verbose=args.verbose)
            tracker.run()

        elif args.mode == 'spacemouse':
            print("Running in Spacemouse mode...")
            spacemouse = VirtualSpacemouse(port=args.port, command_rate=4, verbose=args.verbose)
            spacemouse.start()

        #if args.mode == 'gui':
        #    app = QtWidgets.QApplication(sys.argv)
        #    window = MainWindow(rate=4, verbose=False)
        #    sys.exit(app.exec_())

    except KeyboardInterrupt:
        print("Keyboard interrupt detected...")
        if args.mode == 'spacemouse':
            spacemouse.stop()
        elif args.mode == 'cli':
            tracker.stop()

    finally:
        print("Exiting the application...")
