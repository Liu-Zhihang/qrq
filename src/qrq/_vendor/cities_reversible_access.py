"""7C executable logical-gate reference, not a hardware/QRAM performance model.

Boolean circuits are simulated in bit-sliced basis lanes. The clean subspace
action can then be used for small amplitude simulation without allocating the
exponentially large arithmetic workspace. No hit mask compiles the oracle.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

ZERO, ONE = -1, -2
GATE_KEYS = ("X", "CX", "CCX", "H", "RY", "Z", "MEASURE")


def gate_counts(gates):
    count = Counter(g[0] for g in gates)
    return {key: int(count[key]) for key in GATE_KEYS}


class BooleanBuilder:
    """Fresh-wire XOR/AND network lowered to X, CX and exact Toffoli."""
    def __init__(self, inputs):
        self.inputs = inputs
        self.width = inputs
        self.gates = []
        self.cache = {}

    def node(self, key, controls):
        if key in self.cache:
            return self.cache[key]
        wire = self.width
        self.width += 1
        for op, cs in controls:
            self.gates.append((op, *cs, wire))
        self.cache[key] = wire
        return wire

    def neg(self, a):
        if a == ZERO: return ONE
        if a == ONE: return ZERO
        return self.node(("not", a), [("X", ()), ("CX", (a,))])

    def xor(self, a, b):
        if a == b: return ZERO
        if a == ZERO: return b
        if b == ZERO: return a
        if a == ONE: return self.neg(b)
        if b == ONE: return self.neg(a)
        a, b = sorted((a, b))
        return self.node(("xor", a, b), [("CX", (a,)), ("CX", (b,))])

    def land(self, a, b):
        if a == ZERO or b == ZERO: return ZERO
        if a == ONE: return b
        if b == ONE or a == b: return a
        a, b = sorted((a, b))
        return self.node(("and", a, b), [("CCX", (a, b))])

    def lor(self, a, b):
        return self.xor(self.xor(a, b), self.land(a, b))

    def all(self, bits):
        result = ONE
        for b in bits: result = self.land(result, b)
        return result

    def any(self, bits):
        result = ZERO
        for b in bits: result = self.lor(result, b)
        return result

    def mux(self, select, yes, no):
        return [self.xor(b, self.land(select, self.xor(a, b)))
                for a, b in zip(yes, no)]

    def add(self, a, b, carry=ZERO):
        assert len(a) == len(b)
        output = []
        for x, y in zip(a, b):
            xy = self.xor(x, y)
            output.append(self.xor(xy, carry))
            carry = self.xor(self.land(x, y), self.land(xy, carry))
        return output

    def sub(self, a, b):
        return self.add(a, [self.neg(v) for v in b], ONE)

    def ule(self, a, b):
        # LSB-to-MSB recurrence: at the current most significant differing bit.
        less, equal = ZERO, ONE
        for x, y in zip(a, b):
            same = self.neg(self.xor(x, y))
            less = self.lor(self.land(self.neg(x), y), self.land(same, less))
            equal = self.land(equal, same)
        return self.lor(less, equal)

    def sle(self, a, b):
        assert len(a) == len(b)
        return self.ule(a[:-1]+[self.neg(a[-1])], b[:-1]+[self.neg(b[-1])])

    def signed_times_unsigned(self, signed, unsigned):
        width = len(signed)+len(unsigned)
        ext = signed+[signed[-1]]*(width-len(signed))
        result = [ZERO]*width
        for j, bit in enumerate(unsigned):
            row = [ZERO]*j+[self.land(v, bit) for v in ext[:width-j]]
            result = self.add(result, row)
        return result

    def clean_copy(self, result, target):
        body = self.gates.copy()
        copy = [] if result == ZERO else [("X", target)] if result == ONE else [("CX", result, target)]
        return body+copy+list(reversed(body))


def geometry_circuit(bits):
    """Unsigned coordinates; inputs start[3],end[3],box[lo,hi]*3,valid,flag.

    Signed slab numerators have bits+1 bits, denominators are nonnegative;
    pair products use twice that width. Includes ordered-box rejection.
    """
    if not 1 <= bits <= 32: raise ValueError("coordinate bits outside reference range")
    inputs = 12*bits+2
    b = BooleanBuilder(inputs)
    words = [list(range(i*bits, (i+1)*bits)) for i in range(12)]
    start, end = words[:3], words[3:6]
    valid, flag = inputs-2, inputs-1
    lows, highs, denominators, conditions = [], [], [], [valid]
    n = bits+1
    z, o = [ZERO]*n, [ONE]+[ZERO]*(n-1)
    for axis in range(3):
        p, e = start[axis]+[ZERO], end[axis]+[ZERO]
        lo, hi = words[6+2*axis]+[ZERO], words[7+2*axis]+[ZERO]
        conditions.append(b.ule(lo, hi))
        forward = b.ule(p, e)
        delta = b.mux(forward, b.sub(e, p), b.sub(p, e))
        still = b.neg(b.any(delta))
        low = b.mux(forward, b.sub(lo, p), b.sub(p, hi))
        high = b.mux(forward, b.sub(hi, p), b.sub(p, lo))
        inside = b.land(b.ule(lo, p), b.ule(p, hi))
        conditions.append(b.lor(b.neg(still), inside))
        # A valid stationary axis imposes interval [0,1], avoiding division 0.
        low, high, delta = b.mux(still, z, low), b.mux(still, o, high), b.mux(still, o, delta)
        conditions.extend((b.sle(low, delta), b.sle(z, high)))
        lows.append(low); highs.append(high); denominators.append(delta)
    for i in range(3):
        for j in range(3):
            if i == j: continue  # Ordered bounds already imply low_i <= high_i.
            left = b.signed_times_unsigned(lows[i], denominators[j])
            right = b.signed_times_unsigned(highs[j], denominators[i])
            conditions.append(b.sle(left, right))
    result = b.all(conditions)
    gates = b.clean_copy(result, flag)
    return dict(bits=bits, inputs=inputs, qubits=b.width, flag=flag, valid=valid,
                gates=gates, counts=gate_counts(gates), arithmetic="signed_slab_cross_products")


def pack_words(values, bits):
    if len(values) != 12 or any(int(v) != v or not 0 <= v < (1 << bits) for v in values):
        raise ValueError("twelve in-range integer coordinates required")
    return sum(int(v) << (i*bits) for i, v in enumerate(values))


def execute_lanes(gates, qubits, states):
    """Exact reversible gates on arbitrary integer basis labels, bit sliced."""
    if not states: raise ValueError("empty basis batch")
    lane_count = len(states)
    if any(v < 0 or v.bit_length() > qubits for v in states):
        raise ValueError("basis outside allocated register")
    mask = (1 << lane_count)-1
    wires = [0]*qubits
    active = max(v.bit_length() for v in states)
    for bit in range(active):
        flags = np.fromiter(((s >> bit) & 1 for s in states), dtype=np.uint8, count=lane_count)
        wires[bit] = int.from_bytes(np.packbits(flags, bitorder="little").tobytes(), "little")
    for gate in gates:
        op, *indices = gate
        target = indices[-1]
        if op == "X": wires[target] ^= mask
        elif op == "CX": wires[target] ^= wires[indices[0]]
        elif op == "CCX": wires[target] ^= wires[indices[0]] & wires[indices[1]]
        else: raise ValueError(f"non-Boolean gate {op}")
    return wires


def unpack_lanes(wires, count):
    out = [0]*count
    for bit, lane in enumerate(wires):
        if not lane: continue
        packed = lane.to_bytes((count+7)//8, "little")
        flags = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:count]
        for i in np.flatnonzero(flags): out[int(i)] |= 1 << bit
    return out


def mcx(gates, controls, target, ancillas):
    if not controls: gates.append(("X", target)); return
    if len(controls) == 1: gates.append(("CX", controls[0], target)); return
    if len(controls) == 2: gates.append(("CCX", *controls, target)); return
    if len(ancillas) < len(controls)-2: raise ValueError("insufficient clean ancillas")
    chain = [("CCX", controls[0], controls[1], ancillas[0])]
    for j in range(2, len(controls)-1): chain.append(("CCX", ancillas[j-2], controls[j], ancillas[j-1]))
    gates.extend(chain)
    gates.append(("CCX", ancillas[len(controls)-3], controls[-1], target))
    gates.extend(reversed(chain))


def rom_circuit(records, address_bits, word_bits):
    """Explicit data-only XOR ROM: address + payload/valid + clean selector."""
    if len(records) > 1 << address_bits: raise ValueError("insufficient address")
    if any(x < 0 or x >= 1 << word_bits for x in records): raise ValueError("ROM word overflow")
    ancillas = list(range(address_bits+word_bits, address_bits+word_bits+max(0, address_bits-2)))
    gates = []
    for index, record in enumerate(records):
        if record == 0: continue
        zeros = [j for j in range(address_bits) if not ((index >> j) & 1)]
        gates.extend(("X", j) for j in zeros)
        for bit in range(word_bits):
            if (record >> bit) & 1:
                mcx(gates, list(range(address_bits)), address_bits+bit, ancillas)
        gates.extend(("X", j) for j in reversed(zeros))
    return dict(gates=gates, qubits=address_bits+word_bits+len(ancillas),
                counts=gate_counts(gates), address_bits=address_bits, word_bits=word_bits,
                records=list(records), build_record_words=len(records),
                build_record_bits=len(records)*word_bits)


def compose_oracle(starts, ends, box, bits, geometry):
    """Compile only endpoints and query coordinates, never hit IDs/mask/k."""
    n = len(starts)
    if not n or len(ends) != n: raise ValueError("invalid record arrays")
    address = max(1, (n-1).bit_length())
    capacity = 1 << address
    if any(box[2*a] > box[2*a+1] for a in range(3)): raise ValueError("reversed box")
    for s, e in zip(starts, ends): pack_words(list(s)+list(e)+list(box), bits)
    record_words = [sum(int(v) << (j*bits) for j, v in enumerate(list(s)+list(e))) | (1 << (6*bits))
                    for s, e in zip(starts, ends)]
    rom = rom_circuit(record_words, address, 6*bits+1)
    # Shared geometry register starts after address; the success bit is within it.
    total = address+geometry["qubits"]+max(0, address-2)
    selector_start = address+geometry["qubits"]
    def rom_wire(w):
        if w < address: return w
        v = w-address
        if v < 6*bits: return address+v
        if v == 6*bits: return address+geometry["valid"]
        return selector_start+v-(6*bits+1)
    load = [(g[0], *(rom_wire(w) for w in g[1:])) for g in rom["gates"]]
    q_init = [("X", address+(6+j)*bits+b) for j, v in enumerate(box)
              for b in range(bits) if (int(v) >> b) & 1]
    geom = [(g[0], *(address+w for w in g[1:])) for g in geometry["gates"]]
    gates = load+q_init+geom+list(reversed(q_init))+list(reversed(load))
    flag = address+geometry["flag"]
    inputs = [i | (f << flag) for f in (0, 1) for i in range(capacity)]
    wires = execute_lanes(gates, total, inputs)
    outputs = unpack_lanes(wires, len(inputs))
    marked = []
    for i, value in enumerate(outputs):
        original = inputs[i]
        if (value ^ original) & ~(1 << flag): raise AssertionError("oracle garbage or changed ID")
        marked.append((value >> flag) & 1)
    if marked[capacity:] != [1-v for v in marked[:capacity]]:
        raise AssertionError("oracle is not a clean XOR")
    return dict(address_bits=address, capacity=capacity, qubits=total, flag=flag,
                gates=gates, counts=gate_counts(gates), rom=rom,
                induced_mask=marked[:capacity], basis_inputs=inputs, basis_outputs=outputs,
                query_initialization_X=len(q_init), all_workspace_clean=True)


def canonical_ranges(ids):
    ids = sorted(map(int, ids))
    if len(set(ids)) != len(ids) or (ids and ids[0] < 0): raise ValueError("non-injective IDs")
    ranges = []
    for i in ids:
        if ranges and ranges[-1][1] == i: ranges[-1][1] = i+1
        else: ranges.append([i, i+1])
    return ranges


def validate_ranges(ranges, address_bits):
    previous = 0
    for j, (lo, hi) in enumerate(ranges):
        if not 0 <= lo < hi <= (1 << address_bits) or (j and lo < previous):
            raise ValueError("invalid/overlapping ranges")
        previous = hi


def prepare_circuit(ranges, address_bits):
    """Prefix-count rotations compiled from disjoint intervals without ID expansion."""
    validate_ranges(ranges, address_bits)
    total = sum(hi-lo for lo, hi in ranges)
    if not total: raise ValueError("empty preparation")
    ancillas = list(range(address_bits, address_bits+max(0, address_bits-1)))
    gates, decisions = [], []
    metadata_ops = 0
    def count(lo, hi):
        nonlocal metadata_ops
        metadata_ops += len(ranges)
        return sum(max(0, min(hi, b)-max(lo, a)) for a, b in ranges)
    if total == 1 << address_bits:
        gates = [("H", j) for j in range(address_bits)]
    else:
        def visit(prefix, depth):
            if depth == address_bits: return
            target = address_bits-depth-1
            lo, hi = prefix << (target+1), (prefix+1) << (target+1)
            left, right = count(lo, (lo+hi)//2), count((lo+hi)//2, hi)
            if not left+right: return
            controls = list(range(address_bits-1, target, -1))
            zeros = [w for j, w in enumerate(controls) if not ((prefix >> (depth-1-j)) & 1)]
            theta = 2*math.atan2(math.sqrt(right), math.sqrt(left))
            decisions.append(dict(prefix=prefix, depth=depth, left=left, right=right, theta=theta))
            if right:
                gates.extend(("X", w) for w in zeros)
                if not left:
                    mcx(gates, controls, target, ancillas)
                elif not controls:
                    gates.append(("RY", target, theta))
                else:
                    # Compute all controls to a clean selector; exact CRy decomposition.
                    selector = controls[0] if len(controls) == 1 else ancillas[-1]
                    before = []
                    if len(controls) > 1: mcx(before, controls, selector, ancillas[:-1])
                    gates.extend(before)
                    gates.extend((("RY", target, theta/2), ("CX", selector, target),
                                  ("RY", target, -theta/2), ("CX", selector, target)))
                    gates.extend(reversed(before))
                gates.extend(("X", w) for w in reversed(zeros))
            if left: visit(prefix*2, depth+1)
            if right: visit(prefix*2+1, depth+1)
        visit(0, 0)
    used = [w for g in gates for w in (g[1:2] if g[0] == "RY" else g[1:])]
    return dict(ranges=[list(v) for v in ranges], domain=total, address_bits=address_bits,
                qubits=max([address_bits]+[w+1 for w in used]), gates=gates, counts=gate_counts(gates),
                decisions=decisions, classical_interval_count_operations=metadata_ops,
                classical_rotation_angles=sum(g[0] == "RY" for g in gates))


def apply_state(gates, state, inverse=False):
    out = np.asarray(state, dtype=complex).copy()
    if out.size & (out.size-1): raise ValueError("state size not a power of two")
    if out.size > 1 << 22: raise ValueError("full state simulation cap exceeded")
    indices = np.arange(len(out))
    for g in (list(reversed(gates)) if inverse else gates):
        op = g[0]
        if op in ("X", "CX", "CCX"):
            target = g[-1]
            chosen = (indices & (1 << target)) == 0
            for c in g[1:-1]: chosen &= (indices & (1 << c)) != 0
            lo = indices[chosen]; hi = lo | (1 << target)
            a = out[lo].copy(); out[lo] = out[hi]; out[hi] = a
        elif op in ("H", "RY"):
            target = g[1]
            lo = indices[(indices & (1 << target)) == 0]; hi = lo | (1 << target)
            a, b = out[lo].copy(), out[hi].copy()
            if op == "H": out[lo], out[hi] = (a+b)/math.sqrt(2), (a-b)/math.sqrt(2)
            else:
                theta = -g[2] if inverse else g[2]
                c, s = math.cos(theta/2), math.sin(theta/2)
                out[lo], out[hi] = c*a-s*b, s*a+c*b
        elif op == "Z": out[(indices & (1 << g[1])) != 0] *= -1
        else: raise ValueError(f"unsupported gate {op}")
    return out


class CompiledPreparation:
    """7B-compatible adapter using actual gate simulation, including clean ancilla."""
    def __init__(self, circuit):
        self.circuit = circuit
        self.capacity = 1 << circuit["address_bits"]
        zero = np.zeros(1 << circuit["qubits"], complex); zero[0] = 1
        output = apply_state(circuit["gates"], zero)
        if np.linalg.norm(output[self.capacity:]) > 1e-10: raise AssertionError("prep garbage")
        self.target = output[:self.capacity]

    def _apply(self, state, inverse=False):
        out = np.zeros_like(state, dtype=complex)
        for col in range(state.shape[1]):
            extended = np.zeros(1 << self.circuit["qubits"], complex)
            extended[:self.capacity] = state[:, col]
            result = apply_state(self.circuit["gates"], extended, inverse)
            if np.linalg.norm(result[self.capacity:]) > 1e-10: raise AssertionError("prep dirty subspace")
            out[:, col] = result[:self.capacity]
        return out

    def apply(self, state): return self._apply(state)
    def inverse(self, state): return self._apply(state, True)


def zero_reflection_circuit(address_bits):
    # Phase minus on the all-zero ID, clean ancillas used for MCX lowering.
    gates = [("X", j) for j in range(address_bits)]
    target = address_bits-1
    gates.append(("H", target))
    anc = list(range(address_bits, address_bits+max(0, address_bits-3)))
    mcx(gates, list(range(address_bits-1)), target, anc)
    gates.append(("H", target))
    gates.extend(("X", j) for j in reversed(range(address_bits)))
    return dict(gates=gates, qubits=address_bits+len(anc), counts=gate_counts(gates))


def exact_hit(start, end, box):
    """Classical face-event reference, never used to compile a quantum oracle."""
    delta = [int(e)-int(s) for s, e in zip(start, end)]
    times = {Fraction(0), Fraction(1)}
    for a, d in enumerate(delta):
        if d:
            for face in box[2*a:2*a+2]:
                t = Fraction(int(face)-int(start[a]), d)
                if 0 <= t <= 1: times.add(t)
    return any(all(box[2*a] <= start[a]+t*delta[a] <= box[2*a+1] for a in range(3)) for t in times)


@dataclass
class RangeNode:
    lo: int
    hi: int
    lower: tuple
    upper: tuple
    children: tuple


class IDRangeIndex:
    """Small contiguous-ID AABB reference, NOT a new production spatial index."""
    def __init__(self, starts, ends):
        if not starts or len(starts) != len(ends): raise ValueError("nonempty paired segments required")
        self.starts, self.ends = starts, ends
        self.nodes = 0
        def build(lo, hi):
            self.nodes += 1
            if hi-lo == 1:
                lower = tuple(min(starts[lo][a], ends[lo][a]) for a in range(3))
                upper = tuple(max(starts[lo][a], ends[lo][a]) for a in range(3))
                return RangeNode(lo, hi, lower, upper, ())
            middle = (lo+hi)//2
            children = (build(lo, middle), build(middle, hi))
            return RangeNode(lo, hi, tuple(min(c.lower[a] for c in children) for a in range(3)),
                             tuple(max(c.upper[a] for c in children) for a in range(3)), children)
        self.root = build(0, len(starts))

    def query(self, box, mode, handoff_depth=None):
        if mode not in ("candidate", "enumerate", "blocks"): raise ValueError("unknown mode")
        if mode == "candidate" and (handoff_depth is None or handoff_depth < 0): raise ValueError("depth required")
        stats = dict(node_reads=0, bound_checks=0, geometry_tests=0,
                     materialized_ids=0, range_descriptors=0, contained_shortcuts=0)
        ranges, trace, materialized = [], [], []
        def visit(node, depth):
            stats["node_reads"] += 1; stats["bound_checks"] += 1
            disjoint = any(node.upper[a] < box[2*a] or node.lower[a] > box[2*a+1] for a in range(3))
            if disjoint:
                trace.append([node.lo, node.hi, depth, "disjoint"]); return
            stats["bound_checks"] += 1
            contained = all(box[2*a] <= node.lower[a] and node.upper[a] <= box[2*a+1] for a in range(3))
            if mode == "candidate" and (contained or depth >= handoff_depth or not node.children):
                ranges.append([node.lo, node.hi]); trace.append([node.lo, node.hi, depth, "candidate"]); return
            if mode == "blocks" and contained:
                ranges.append([node.lo, node.hi]); stats["contained_shortcuts"] += 1
                trace.append([node.lo, node.hi, depth, "contained"]); return
            if not node.children:
                stats["geometry_tests"] += 1
                hit = exact_hit(self.starts[node.lo], self.ends[node.lo], box)
                if hit:
                    ranges.append([node.lo, node.hi])
                    if mode == "enumerate":
                        stats["materialized_ids"] += 1
                        materialized.append(node.lo)
                trace.append([node.lo, node.hi, depth, "hit" if hit else "miss"])
            else:
                trace.append([node.lo, node.hi, depth, "descend"])
                for c in node.children: visit(c, depth+1)
        visit(self.root, 0)
        stats["range_descriptors"] = len(ranges)
        return dict(ranges=ranges, stats=stats, trace=trace, mode=mode, handoff_depth=handoff_depth,
                    materialized_answer_ids=materialized if mode == "enumerate" else None)


def initialization_requirements(address_bits, peak_qubits, expected, classical):
    """Zero-state requirements, NOT a decomposition or an assumed free reset.

    Every reached attempt after the first needs a fresh/reset ID register.
    Oracle work is already clean; the measured failure flag is already zero.
    A final failed query returns no state and need not reset again inside it.
    """
    attempts = expected.get("attempts", 0.0)
    return dict(initial_zero_qubits=int(peak_qubits),
                expected_retry_reset_qubits=0.0 if classical else address_bits*max(0.0, attempts-1),
                reset_physical_decomposition="unspecified_nonzero_unit_cost_not_assumed")


class GateDerivedMark:
    """Induced clean subspace action obtained by executing the full gate program."""
    def __init__(self, oracle):
        if not oracle["all_workspace_clean"]: raise ValueError("unverified workspace")
        self.mask = np.asarray(oracle["induced_mask"], bool)

    def compute(self, state):
        out = state.copy()
        out[self.mask] = out[self.mask, ::-1]
        return out


def compiled_branches(prep, mark, max_window):
    from cities_coherent_semantics import Attempt
    state = np.zeros((prep.capacity, 2), complex); state[0, 0] = 1
    state = prep.apply(state)
    reflection = zero_reflection_circuit(prep.circuit["address_bits"])
    branches, worst = [], 0.0
    for r in range(max_window):
        measured = mark.compute(state)
        good = measured[:, 1].copy()
        probability = float(np.vdot(good, good).real)
        branches.append(Attempt(r, probability, good, measured, worst,
                                abs(float(np.vdot(measured, measured).real)-1)))
        # X H |0> = |->, one clean U, H X resets the phase flag.
        state = state[:, ::-1].copy()
        state = np.column_stack((state[:, 0]+state[:, 1], state[:, 0]-state[:, 1]))/math.sqrt(2)
        state = mark.compute(state)
        state = np.column_stack((state[:, 0]+state[:, 1], state[:, 0]-state[:, 1]))/math.sqrt(2)
        state = state[:, ::-1].copy()
        worst = max(worst, float(np.vdot(state[:, 1], state[:, 1]).real))
        state = prep.inverse(state)
        for col in range(2):
            extended = np.zeros(1 << reflection["qubits"], complex)
            extended[:prep.capacity] = state[:, col]
            reflected = apply_state(reflection["gates"], extended)
            if np.linalg.norm(reflected[prep.capacity:]) > 1e-10: raise AssertionError("reflection garbage")
            state[:, col] = reflected[:prep.capacity]
        state = -prep.apply(state)
    return branches


def logical_resources(preparation, oracle_counts, expected, route, phase_profile="kickback"):
    if phase_profile not in ("kickback", "compute_uncompute"): raise ValueError("phase profile")
    result = {g: 0.0 for g in GATE_KEYS}
    if not expected: return result
    r = expected.get("zero_reflections", 0.0)
    attempts = expected.get("attempts", 0.0)
    a_calls = expected.get("prepare_forward", 0)+expected.get("prepare_inverse", 0)
    marks = 0 if route.startswith("C_") else attempts+r*(1 if phase_profile == "kickback" else 2)
    reflection = zero_reflection_circuit(preparation["address_bits"])["counts"]
    for g in GATE_KEYS:
        result[g] = a_calls*preparation["counts"][g]+marks*oracle_counts[g]+r*reflection[g]
    if not route.startswith("C_"):
        if phase_profile == "kickback": result["X"] += 2*r; result["H"] += 2*r
        else: result["Z"] += r
        result["X"] += expected["successful_flag_resets"]
        result["MEASURE"] += attempts
    return result
