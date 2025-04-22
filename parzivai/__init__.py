try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore
from parzivai.input_output import get_vectorstore

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata

__all__ = [
    "get_vectorstore",
]
