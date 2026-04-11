__all__ = ["LSLClient"]


def __getattr__(name):
	if name == "LSLClient":
		from ._lsl_client import LSLClient

		return LSLClient
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")