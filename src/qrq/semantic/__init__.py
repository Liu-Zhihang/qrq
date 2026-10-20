"""EXPERIMENTAL extension point for semantic predicates.

Nothing in this sub-package has a matched cost model yet. It fixes the
interfaces a future semantic query layer should implement:

* :class:`SemanticEncoder` maps labels (land use, activity type, ...) onto
  one ordered numeric axis so that a label set becomes an interval;
* :class:`CategoricalIntervalEncoder` is a toy encoder over a fixed order;
* :class:`SemanticBox` extends :class:`qrq.geometry.SpaceTimeBox` with one
  closed semantic interval.

The shipped indexes are three-dimensional; a semantic box is answered by a
3-D index on its space-time part followed by a semantic filter. A native
4-D index and a matched cost model for semantic predicates are future work
(see ``docs/roadmap.md``). These interfaces may change without notice.
"""
from .box import SemanticBox
from .categorical import CategoricalIntervalEncoder
from .protocols import SemanticEncoder

__all__ = ["CategoricalIntervalEncoder", "SemanticBox", "SemanticEncoder"]
