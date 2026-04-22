"""Alignment: cross-modal temporal alignment — TBD.

This module is a stub. Full design is deferred to Phase 5+.

Future responsibilities:
- Align EEG with audio/video stimuli at sub-sample precision
- Handle different sampling rates between modalities
- Compensate for trigger-to-stimulus latency
- Support dynamic time warping for speech alignment

.. warning::
    Not yet implemented. All functions in this module raise
    ``NotImplementedError`` with a descriptive message.
"""


def align_modalities(*args, **kwargs):
    """Align two modality signals in time. Not yet implemented."""
    raise NotImplementedError(
        "Cross-modal temporal alignment is not yet implemented (planned Phase 5+). "
        "For paired multi-modal data, use MultiModalAssembler with run-level pairing."
    )


def resample_to_common_rate(*args, **kwargs):
    """Resample multiple modalities to a common sampling rate. Not yet implemented."""
    raise NotImplementedError(
        "Cross-modal resampling is not yet implemented (planned Phase 5+)."
    )
