__all__ = ["launch_landmark_selector"]


def __getattr__(name):
	if name == "launch_landmark_selector":
		from ._landmark_selector import launch_landmark_selector

		return launch_landmark_selector
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")