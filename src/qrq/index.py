"""Static 3-D record indexes and their matched access plans.

:class:`SpatioTemporalIndex` wraps the compiled reference backend
``qrq._qrq_matched.QuantumTrajectoryTree3D``. Points are passed to the
backend as zero-length segments (``starts == ends``), exactly as in the
sealed workloads. Three bulk-loaded tree layouts are available:

``packed_rtree``
    Sort-tile-recursive packing on node centres (x, then y, then t).
``tb_tree``
    Trajectory-bundled leaves: records are grouped by entity id first.
``kd_tree``
    Median splits on the widest centre axis.

:meth:`SpatioTemporalIndex.plan` returns an :class:`AccessPlan` that mirrors
the backend ``matched_query`` dictionary field by field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .geometry import SpaceTimeBox
from .records import backend_arrays

TREE_KINDS: tuple[str, ...] = ("packed_rtree", "tb_tree", "kd_tree")
"""Tree layouts, in the order used for envelope tie-breaking."""

DEFAULT_FANOUT = 4
"""Branching factor of every published index."""

Range = tuple[int, int]


def _load_backend() -> Any:
    try:
        from . import _qrq_matched
    except ImportError as exc:  # pragma: no cover - depends on the build
        raise ImportError(
            "the compiled backend qrq._qrq_matched is not available; "
            "install qrq with 'pip install .' so the extension is built"
        ) from exc
    return _qrq_matched


@dataclass(frozen=True)
class ClassicalAccess:
    """Work and answer ranges of one exact classical traversal.

    ``node_reads``, ``slots`` and ``geometry`` count node visits, entry-slot
    scans and exact record tests. ``ranges`` are half-open rank intervals of
    the answer in the static depth-first rank layout.
    """

    node_reads: int
    slots: int
    geometry: int
    ranges: tuple[Range, ...]

    @classmethod
    def from_matched(cls, block: Mapping[str, Any]) -> "ClassicalAccess":
        return cls(
            int(block["node_reads"]),
            int(block["slots"]),
            int(block["geometry"]),
            tuple((int(lo), int(hi)) for lo, hi in block["ranges"]),
        )

    def to_matched(self) -> dict[str, Any]:
        return dict(
            node_reads=self.node_reads,
            slots=self.slots,
            geometry=self.geometry,
            ranges=list(self.ranges),
        )


@dataclass(frozen=True)
class AccessPlan:
    """Matched access plan of one box on one index.

    Fields mirror the backend ``matched_query`` result:

    * ``candidate_ranges``: rank intervals of the first-witness frontier
      (the candidate domain prepared by the quantum route);
    * ``prefix_nodes``, ``prefix_slots``, ``prefix_geometry``: classical
      work spent discovering that frontier;
    * ``frontier_size``, ``height``, ``depth``, ``witness_kind``: frontier
      description (``depth == -1`` and ``witness_kind == 'none'`` when the
      box is empty);
    * ``block``: exact traversal that stops at fully contained subtrees;
    * ``enumeration``: exact traversal that tests every reachable record.
    """

    candidate_ranges: tuple[Range, ...]
    prefix_nodes: int
    prefix_slots: int
    prefix_geometry: int
    frontier_size: int
    height: int
    depth: int
    witness_kind: str
    block: ClassicalAccess
    enumeration: ClassicalAccess

    @classmethod
    def from_matched(cls, plan: Mapping[str, Any]) -> "AccessPlan":
        """Convert a backend ``matched_query`` dictionary."""
        return cls(
            candidate_ranges=tuple((int(lo), int(hi)) for lo, hi in plan["candidate_ranges"]),
            prefix_nodes=int(plan["prefix_nodes"]),
            prefix_slots=int(plan["prefix_slots"]),
            prefix_geometry=int(plan["prefix_geometry"]),
            frontier_size=int(plan["frontier_size"]),
            height=int(plan["height"]),
            depth=int(plan["depth"]),
            witness_kind=str(plan["witness_kind"]),
            block=ClassicalAccess.from_matched(plan["block"]),
            enumeration=ClassicalAccess.from_matched(plan["enumeration"]),
        )

    def to_matched(self) -> dict[str, Any]:
        """Return the plan in the backend ``matched_query`` dictionary form."""
        return dict(
            candidate_ranges=list(self.candidate_ranges),
            prefix_nodes=self.prefix_nodes,
            prefix_slots=self.prefix_slots,
            prefix_geometry=self.prefix_geometry,
            frontier_size=self.frontier_size,
            height=self.height,
            depth=self.depth,
            witness_kind=self.witness_kind,
            block=self.block.to_matched(),
            enumeration=self.enumeration.to_matched(),
        )


class SpatioTemporalIndex:
    """A static bulk-loaded index over point records.

    Use :meth:`build` to construct one. The index keeps a reference to the
    ``(N, 3)`` coordinate array so that answers can be verified by brute
    force (:meth:`count`).
    """

    def __init__(self, tree: Any, kind: str, fanout: int, xyz: np.ndarray, entity: np.ndarray):
        self._tree = tree
        self._kind = kind
        self._fanout = fanout
        self._xyz = xyz
        self._entity = entity

    @classmethod
    def build(
        cls,
        points: np.ndarray,
        ids: np.ndarray | None = None,
        kind: str = "packed_rtree",
        fanout: int = DEFAULT_FANOUT,
    ) -> "SpatioTemporalIndex":
        """Bulk-load an index.

        ``points`` is a :data:`qrq.records.POINT` array (``ids`` defaults to
        its ``entity`` field) or an ``(N, 3)`` float64 ``(x, y, t)`` array
        with an explicit int64 ``ids`` array. ``kind`` is one of
        :data:`TREE_KINDS`; ``fanout`` must be a power of two >= 2.
        The backend rejects empty inputs.
        """
        if kind not in TREE_KINDS:
            raise ValueError(f"kind must be one of {TREE_KINDS}")
        xyz, entity = backend_arrays(points, ids)
        if len(xyz) == 0:
            raise ValueError("an index needs at least one record")
        backend = _load_backend()
        tree = backend.QuantumTrajectoryTree3D(xyz, xyz, entity, int(fanout), kind)
        return cls(tree, kind, int(fanout), xyz, entity)

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def fanout(self) -> int:
        return self._fanout

    @property
    def size(self) -> int:
        """Number of indexed records ``n``."""
        return int(self._tree.size)

    @property
    def height(self) -> int:
        return int(self._tree.height)

    @property
    def node_count(self) -> int:
        return int(self._tree.node_count)

    @property
    def points(self) -> np.ndarray:
        """Read-only view of the indexed ``(N, 3)`` coordinates."""
        view = self._xyz.view()
        view.flags.writeable = False
        return view

    def plan(self, box: SpaceTimeBox) -> AccessPlan:
        """Return the matched access plan of ``box``."""
        return AccessPlan.from_matched(self.matched_query(box))

    def matched_query(self, box: SpaceTimeBox) -> dict[str, Any]:
        """Return the raw backend ``matched_query`` dictionary of ``box``."""
        return dict(self._tree.matched_query(*box.bounds))

    def rank_to_record(self) -> np.ndarray:
        """Record index stored at each rank of the static rank layout."""
        return np.asarray(self._tree.matched_layout()["rank_to_id"], dtype=np.int64)

    def count(self, box: SpaceTimeBox) -> int:
        """Brute-force number of indexed records inside ``box``."""
        return box.count(self._xyz)
