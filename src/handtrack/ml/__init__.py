__all__ = ["EMGRegressor", "ModelManager"]


def __getattr__(name):
	if name == "EMGRegressor":
		from ._models import EMGRegressor

		return EMGRegressor
	if name == "ModelManager":
		from ._model_manager import ModelManager

		return ModelManager
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")