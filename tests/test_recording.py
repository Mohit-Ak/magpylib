"""The recording hook: top-level calls only, arguments as written, resolved."""

import numpy as np

import magpylib as magpy
from magpylib._src.recording import record


class Handle:
    """A value only the recorder can turn into a number."""

    def __init__(self, value):
        self.value = value


def _resolve(value):
    return value.value if isinstance(value, Handle) else value


def test_records_top_level_calls_only():
    log = []
    with record(log.append):
        m = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
        m.rotate_from_angax(30, "z", anchor=0)
        m.position = (2, 0, 0)
        c = magpy.Collection(m)
        c.move((0, 0, 1))
        m2 = m.copy(position=(3, 0, 0))
    assert [(e.kind, e.name) for e in log] == [
        ("new", "Cuboid"),
        ("call", "rotate_from_angax"),
        ("set", "position"),
        ("new", "Collection"),
        ("call", "move"),
        ("call", "copy"),
    ]
    assert log[-1].result is m2
    assert log[3].args == (m,)


def test_arguments_are_kept_as_written_and_resolved_for_magpylib():
    log = []
    r = Handle(0.02)
    with record(log.append, resolve=_resolve):
        m = magpy.magnet.Cuboid(
            dimension=(1, 1, 1), polarization=(0, 0, 1), position=(r, 0, 0)
        )
        m.move((0, 0, Handle(0.5)))
    assert log[0].kwargs["position"][0] is r
    assert isinstance(log[1].args[0][2], Handle)
    np.testing.assert_allclose(m.position, (0.02, 0, 0.5))


def test_nothing_is_recorded_outside_the_block():
    log = []
    with record(log.append):
        pass
    magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1)).move((1, 0, 0))
    assert log == []
