"""CV/YOLO thin facade to existing src.vision module."""
try:
	from src import vision as vision  # реэкспорт существующего модуля
except Exception:  # pragma: no cover
	vision = None  # type: ignore
 
__all__ = ["vision"] 