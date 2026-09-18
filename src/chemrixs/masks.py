"""
stores masks for one detector for one run. all take the same shape of (n_frames,)

Example Use:
--------------

    sel = m.laser_on & m.i0_finite & m.i0_above_floor & (m.mono_bin == i)

m is the mask object. laser_on is a mask stored within it. mono_bin is a mask that stores all of the idx of the bins


THE FRAME AXIS
--------------
the frame axis shoul have length of after "sort + roll" and "countmask" changes
add() enforces the length, so an array accidentally built from an already-sliced source raises
instead of silently misaligning.

`frame_id` holds each surviving frame's position in the pre-countmask array,
so a shot that looks wrong in a bin can be traced back to the raw file.

BOOLEANS AND INDICES
--------------------
Both are just per-shot arrays and both go through add().  A bin index is an
integer array with -1 for out-of-range; compare it where you need it
(`m.mono_bin == i`) rather than storing one mask per bin.
"""

import numpy as np


class Masks:
    """Named per-shot arrays over one fixed frame axis, reachable by dot.

    Parameters
    ----------
    n_frames : int
        Length of the frame axis.  Every array must match it.
    """

    def __init__(self, n_frames):
        self.n = int(n_frames)
        self._arrays = {}
        self._notes = {}

    # ---------------------------------------------------------------- add
    def add(self, name, arr, note=''):
        """Register a per-shot array under `name`, reachable as `m.name`.

        arr : (n_frames,) bool or int
            Boolean for a selection, integer for a bin index.  Anything else
            raises -- a float per-shot quantity belongs on the detector, not
            here.
        note : str
            What this array means, and for a mask what a failing shot
            physically is.  Shown by report(), and the only documentation
            anyone reading this later will have.
        """

        a = np.asarray(arr)
        if a.ndim != 1:
            a = a.squeeze()
        if a.shape != (self.n,):
            raise ValueError(
                f"'{name}' has shape {np.asarray(arr).shape} -> {a.shape}, "
                f"expected ({self.n},).  An array built before the countmask, "
                f"or from an already-sliced source, lands here")
        if a.dtype != bool and not np.issubdtype(a.dtype, np.integer):
            raise TypeError(f"'{name}' is {a.dtype}; masks must be bool and "
                            f"bin indices integer")

        self._arrays[name] = a
        self._notes[name] = note
        return a

    # ------------------------------------------------------------- access
    def __getattr__(self, name):
        # only called when normal lookup fails, and reads __dict__ directly,
        # so it cannot recurse
        try:
            return self.__dict__['_arrays'][name]
        except KeyError:
            raise AttributeError(
                f'{name!r} is not registered; have '
                f'{list(self.__dict__.get("_arrays", {}))}') from None

    def __getitem__(self, name):
        return self._arrays[name]

    def __contains__(self, name):
        return name in self._arrays

    def __len__(self):
        return self.n

    def __dir__(self):
        # so tab-completion in a notebook shows the registered names
        return list(super().__dir__()) + list(self._arrays)

    @property
    def names(self):
        return list(self._arrays)

    def note(self, name):
        return self._notes.get(name, '')

    def to_dict(self):
        """Plain {name: array} for whoever is writing the output file."""
        return dict(self._arrays)

    def notes(self):
        return dict(self._notes)