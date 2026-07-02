"""
integrating_raw.py

Non-destructive variants of the chemrixs SmallData / Integrating classes.

Motivation
----------
chemrixs.integrating.Integrating.countmask() builds a boolean mask from the
per-frame shot count (frames whose `count` deviates from the mode) and applies
it DESTRUCTIVELY to every attribute of every integrating detector. For runs
where a large fraction of frames have a non-modal count, this silently discards
data (e.g. 29182 -> 15716 frames).

A side effect makes the discard easy to misread: summing_channels() calls
delattr() on the raw fim_0 / fim_1 waveform attributes after building I0.
Because those attributes are cached_property-backed, delattr only clears the
cache -- the next access RELOADS the raw, unmasked arrays from the h5 file.
The object then shows a mix of masked (15716) and raw (29182) arrays.

This module provides:

  IntegratingRaw : identical pipeline (get_scanvar -> countmask ->
                   summing_channels), except countmask() only COMPUTES and
                   STORES the mask. Nothing is removed. The per-frame mono
                   energy (encoder -> pitch -> eV via the run-ranged
                   mono_calib in the yaml) is therefore available for EVERY
                   integrated frame.

  SmallDataRaw   : SmallData whose .integrating returns an IntegratingRaw.

Usage
-----
    from integrating_raw import SmallDataRaw

    sd   = SmallDataRaw(path_to_run0120_h5, 'roi_input.yml')  # scantype='mono' to force
    intg = sd.integrating           # runs full pipeline, nothing discarded
    det  = intg.axis_svls

    det.mono.shape        # (N_frames,)  per-frame incident energy, eV, FULL length
    det.full_area.shape   # (N_frames, 796)
    det.countmask         # boolean QA mask (True = frame had the expected count)

    # energy cut on the SVLS map
    E, cut, sel = intg.energy_select(700.0, 702.0)
    spectrum = cut.mean(axis=0)

    # optionally apply the stored mask later
    clean = intg.apply_countmask('axis_svls')   # dict of masked arrays
"""

from functools import cached_property

import numpy as np
import scipy.stats as st

from chemrixs.integrating import Integrating
from chemrixs.smalldata import SmallData


class IntegratingRaw(Integrating):
    """Integrating with a non-destructive countmask.

    The parent __init__ is reused unchanged; Python's dynamic dispatch means
    its internal call to self.countmask() lands here, so the pipeline order
    (get_scanvar -> countmask -> summing_channels) is preserved. get_scanvar
    computes det.mono (eV) at FULL frame length, and summing_channels builds
    fim0/fim1/apd/I0 at FULL frame length, because nothing was removed in
    between.
    """

    # ------------------------------------------------------------------ #
    # overridden: compute + store the mask, do NOT apply it
    # ------------------------------------------------------------------ #
    def countmask(self):
        """Compute the per-detector count mask and store it without applying.

        Stores
        ------
        self.countmasks[detector]      : bool array, True = frame kept by the
                                         original (destructive) logic
        self.expected_counts[detector] : the modal / configured shot count
        det.countmask                  : same bool array, on the detector
        """
        self.countmasks = {}
        self.expected_counts = {}

        for detector, det_spec_dict in self.yaml['int_detectors'].items():
            det = getattr(self, detector)

            if len(self.yaml['expected_count']) == 0:
                expected_count = st.mode(det.count, keepdims=False)[0]
            else:
                expected_count = self.yaml['expected_count']

            mask = (det.count < expected_count + 1) & (det.count > expected_count - 1)
            mask = np.asarray(mask, dtype=bool)

            self.countmasks[detector] = mask
            self.expected_counts[detector] = expected_count
            # attach to the detector for convenience (QA / later weighting)
            setattr(det, 'countmask', mask)

            n_flag = int((~mask).sum())
            print(f"[IntegratingRaw] {detector}: expected count = {expected_count}, "
                  f"{n_flag}/{mask.size} frames flagged by countmask "
                  f"(KEPT -- no frames removed)")

    # ------------------------------------------------------------------ #
    # optional: apply the stored mask on demand, non-destructively
    # ------------------------------------------------------------------ #
    def apply_countmask(self, detector='axis_svls', attrs=None):
        """Return count-masked copies of the requested attributes.

        Does NOT modify the detector object. Attributes whose leading
        dimension does not match the mask length are skipped with a warning
        (this is how the fim length mismatch surfaces instead of crashing).

        Parameters
        ----------
        detector : str
            Which integrating detector's mask/attributes to use.
        attrs : list of str or None
            Attribute names to mask. Default: everything in the detector's
            attrdict plus derived quantities (mono, delay, I0, apd, fim0,
            fim1) when present.

        Returns
        -------
        dict : {attr_name: masked_array}
        """
        det = getattr(self, detector)
        mask = self.countmasks[detector]
        det_spec_dict = self.yaml['int_detectors'][detector]

        if attrs is None:
            attrs = list(self.yaml[det_spec_dict['attrdict']].keys())
            for extra in ('mono', 'delay', 'I0', 'apd', 'fim0', 'fim1'):
                if hasattr(det, extra) and extra not in attrs:
                    attrs.append(extra)

        out = {}
        for at in attrs:
            if not hasattr(det, at):
                continue
            a = getattr(det, at)
            lead = getattr(a, 'shape', (None,))[0]
            if lead != mask.size:
                print(f"[IntegratingRaw] skipping '{at}': leading dim {lead} "
                      f"!= mask length {mask.size}")
                continue
            out[at] = a[mask]
        return out

    # ------------------------------------------------------------------ #
    # convenience: slice frames by calibrated incident energy
    # ------------------------------------------------------------------ #
    def energy_select(self, emin, emax, detector='axis_svls', attr='full_area',
                      require_good_count=False):
        """Select integrated frames whose incident (mono) energy lies in
        [emin, emax] and return the matching detector data.

        Frames with count == 0 produce non-finite mono values (encoder_sum /
        count); those are always excluded, since they have no defined energy.

        Parameters
        ----------
        emin, emax : float
            Energy window in eV (inclusive).
        detector : str
            Integrating detector name (default 'axis_svls').
        attr : str
            Which attribute to slice (default 'full_area').
        require_good_count : bool
            If True, additionally require the stored countmask (i.e. only
            frames the original pipeline would have kept).

        Returns
        -------
        E    : (M,) energies of the selected frames
        data : (M, ...) selected rows of `attr`
        sel  : (N,) boolean selector over ALL frames (for reuse on other
               attributes, e.g. I0 for normalisation)
        """
        det = getattr(self, detector)
        E = np.asarray(det.mono, dtype=float)

        sel = np.isfinite(E) & (E >= emin) & (E <= emax)
        if require_good_count:
            sel &= self.countmasks[detector]

        data = getattr(det, attr)
        if getattr(data, 'shape', (None,))[0] != E.size:
            raise ValueError(
                f"'{attr}' leading dimension {getattr(data, 'shape', None)} "
                f"does not match number of frames {E.size}; cannot slice by "
                f"energy. (Raw waveform attrs reloaded from h5 may have a "
                f"different frame count.)")

        return E[sel], data[sel], sel


class SmallDataRaw(SmallData):
    """SmallData whose `integrating` property builds an IntegratingRaw."""

    @cached_property
    def integrating(self):
        if not self.is_open():
            self.open()
        # parent stores the group under name-mangled attribute
        intgrp = self._SmallData__intgrp
        return IntegratingRaw(intgrp, self.yaml, self.epics,
                              self.scantype, self.run)