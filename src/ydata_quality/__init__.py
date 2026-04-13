"""
YData open-source lib for Data Quality.
"""
from warnings import warn

from .core.data_quality import DataQuality

from .__version__ import __version__  # noqa: F401

warn(
    """
    `import ydata_quality` is deprecated and will not receive more updates. 
    Please install data-quality via `pip install fg-data-quality` and use `import data_quality` instead.
    """,
    DeprecationWarning,
    stacklevel=2,
)


__all__ = [
    "DataQuality"
]
