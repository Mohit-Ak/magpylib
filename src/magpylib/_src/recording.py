"""Record the calls a script makes on Magpylib objects.

Inside a ``record`` block, every top-level construction, property assignment
and transformation is reported to a callback with the arguments it was given,
before Magpylib turns them into numbers. Calls Magpylib makes on itself while
doing that work (a constructor setting its properties, a collection moving its
children) are not reported.

An argument may be a value that only the recorder knows how to turn into a
number, such as a variable in a parametric document. The recorder's
``resolve`` turns each such value into a number before Magpylib sees it, so
Magpylib computes with plain numbers while the record keeps what was written.
"""

import functools
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

_RECORDER = ContextVar("magpylib_recorder", default=None)
_INSIDE = ContextVar("magpylib_recorder_inside", default=False)

# Public methods that change an object, by name; wrapped where a class defines them.
RECORDED_METHODS = (
    "move",
    "rotate",
    "rotate_from_angax",
    "rotate_from_rotvec",
    "rotate_from_euler",
    "rotate_from_matrix",
    "rotate_from_mrp",
    "rotate_from_quat",
    "copy",
    "add",
    "remove",
    "reset_path",
)


@dataclass(frozen=True)
class Event:
    """One recorded call.

    Parameters
    ----------
    kind : {"new", "set", "call"}
        A construction, a property assignment, or a method call.
    obj : object
        The object constructed, assigned to, or called on.
    name : str
        The class name for "new", else the property or method name.
    args, kwargs : tuple, dict
        The arguments as written, before ``resolve``.
    result : object
        What the call returned (the new object for ``copy``).
    """

    kind: str
    obj: Any
    name: str
    args: tuple
    kwargs: dict
    result: Any = None


class record:
    """Report Magpylib calls made inside the block to ``on_event``.

    Parameters
    ----------
    on_event : callable
        Called with each ``Event``, after the call has run.
    resolve : callable or None
        Called with every argument leaf (inside tuples, lists and dicts) before
        Magpylib sees it; returns what Magpylib should get instead. ``None``
        passes arguments through unchanged.
    """

    def __init__(self, on_event, resolve=None):
        self.on_event = on_event
        self.resolve = resolve
        self._token = None

    def __enter__(self):
        self._token = _RECORDER.set(self)
        return self

    def __exit__(self, *exc):
        _RECORDER.reset(self._token)

    def _resolved(self, value):
        if self.resolve is None:
            return value
        if isinstance(value, tuple | list):
            return type(value)(self._resolved(v) for v in value)
        if isinstance(value, dict):
            return {k: self._resolved(v) for k, v in value.items()}
        return self.resolve(value)


def _recorded(fn, kind, name=None):
    """Wrap ``fn(self, *args, **kwargs)`` so a top-level call is recorded."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        recorder = _RECORDER.get()
        if recorder is None or _INSIDE.get():
            return fn(self, *args, **kwargs)
        token = _INSIDE.set(True)
        try:
            result = fn(self, *recorder._resolved(args), **recorder._resolved(kwargs))
        finally:
            _INSIDE.reset(token)
        if kind == "new":
            label = magpylib_class(type(self)).__name__
        else:
            label = name or fn.__name__
        recorder.on_event(Event(kind, self, label, args, kwargs, result))
        return result

    wrapper._magpylib_recorded = True
    return wrapper


def magpylib_class(cls):
    """The nearest class in ``cls``'s hierarchy that Magpylib defines."""
    return next(c for c in cls.__mro__ if c.__module__.startswith("magpylib."))


def wrap_class(cls):
    """Wrap, in place and once, what ``cls`` itself defines: ``__init__``,
    property setters, and the methods in ``RECORDED_METHODS``.

    Only Magpylib's own classes: a subclass written elsewhere is transparent,
    so the Magpylib calls its methods make are what gets recorded, and its
    objects are reported as the Magpylib class they extend.
    """
    if cls.__dict__.get("_magpylib_recording_wrapped"):
        return
    if not cls.__module__.startswith("magpylib."):
        return
    for attr, value in list(cls.__dict__.items()):
        if attr == "__init__" and callable(value):
            setattr(cls, attr, _recorded(value, "new"))
        elif isinstance(value, property) and value.fset and not attr.startswith("_"):
            fset = _recorded(value.fset, "set", attr)
            setattr(cls, attr, property(value.fget, fset, value.fdel, value.__doc__))
        elif attr in RECORDED_METHODS and callable(value):
            setattr(cls, attr, _recorded(value, "call"))
    cls._magpylib_recording_wrapped = True
