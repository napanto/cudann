"""cudann - CUDA feed-forward neural network library.

The compiled module ``_cudann`` is re-exported as is (``Network_double``,
``Network_float``, the ``*_double`` / ``*_float`` configuration classes and the
enums, exactly as in syclnn 0.1).  On top of that:

* :func:`devices` / :func:`build_info` / :class:`Options` (identical to syclnn)
* :func:`Network` - a convenience factory taking ``dtype`` and keyword options.
"""

from importlib import import_module as _imp

_cudann = _imp("._cudann", package=__name__)
globals().update({k: v for k, v in _cudann.__dict__.items() if not k.startswith("__")})
__version__ = _cudann.__version__
__all__ = [k for k in _cudann.__dict__ if not k.startswith("_")] + ["Network", "make_options"]

_DTYPE_SUFFIX = {"double": "double", "float64": "double", "float": "float", "float32": "float"}

# 0.1 exported the members of the *_double enums into the module namespace
# (cudann.L2, cudann.Adam, ...); keep those aliases for old notebooks.
for _enum in ("RegularizationType_double", "StopCriteriaType_double", "MomentumType_double",
              "AdaptiveLearningStrategy_double"):
    for _member in getattr(_cudann, _enum):
        globals().setdefault(_member.name, _member)
        if _member.name not in __all__:
            __all__.append(_member.name)
del _enum, _member


def make_options(**kwargs):
    """Build an :class:`Options` from keyword arguments (raises on unknown names)."""
    opts = _cudann.Options()
    for key, value in kwargs.items():
        if not hasattr(opts, key):
            raise AttributeError(f"Options has no attribute {key!r}")
        setattr(opts, key, value)
    return opts


def Network(layers, learning_rate, *args, dtype="double", options=None, **kwargs):
    """Factory: ``Network(layers, lr, ..., dtype="double"|"float", device="cpu", profile=True, ...)``.

    Positional/keyword arguments other than ``dtype`` and the option names are
    forwarded to ``Network_<dtype>``; keyword arguments that are ``Options``
    attributes (``device``, ``blas``, ``profile``, ``memory``, ...) build the
    options object.
    """
    suffix = _DTYPE_SUFFIX.get(str(dtype).lower())
    if suffix is None:
        raise ValueError("dtype must be 'double' or 'float'")
    opts = options if options is not None else _cudann.Options()
    probe = _cudann.Options()
    forwarded = {}
    for key, value in kwargs.items():
        if hasattr(probe, key):
            setattr(opts, key, value)
        else:
            forwarded[key] = value
    cls = getattr(_cudann, f"Network_{suffix}")
    return cls(layers, learning_rate, *args, options=opts, **forwarded)
