"""Post-hoc matched-output classical query-table baseline, with paid writes."""
import math
import numpy as np
from .qrq_matched import compile_intervals,simulate_preparation
from .cities_reversible_access import rom_circuit,execute_lanes,unpack_lanes,apply_state


def check_injective_map(labels,n):
    if not labels or len(set(labels))!=len(labels) or not all(0<=i<n for i in labels):raise ValueError('invalid labels')
    bits=max(1,(n-1).bit_length());cap=1<<bits
    inverse=[0]*cap
    for j,i in enumerate(labels):inverse[i]=j
    forward=rom_circuit(labels,bits,bits);back=rom_circuit(inverse,bits,bits)
    def wire(w):return w+bits if w<bits else w-bits if w<2*bits else w
    gates=forward['gates']+[(g[0],*(wire(w) for w in g[1:])) for g in back['gates']]
    qubits=max(forward['qubits'],back['qubits'])
    outputs=unpack_lanes(execute_lanes(gates,qubits,list(range(len(labels)))),len(labels))
    assert outputs==[i<<bits for i in labels]
    prep=compile_intervals([(0,len(labels))],bits,True)
    rank_state,_=simulate_preparation(prep)
    state=np.zeros(1<<qubits,complex);state[:cap]=rank_state
    final=apply_state(gates,state)
    expected=np.zeros_like(final);expected[np.array(labels)<<bits]=1/math.sqrt(len(labels))
    assert np.linalg.norm(final-expected)<1e-10
    return dict(n=n,k=len(labels),bits=bits,checked_basis_states=len(labels),clean=True,
                fidelity=float(abs(np.vdot(expected,final))**2),gate_count=len(gates))


def table_resources(n,k,enum_row,write_units=4):
    bits=max(1,(n-1).bit_length())
    prep=compile_intervals([(0,k)] if k else [],bits)
    c=enum_row['classical']
    return dict(route='C_TABLE',domain=k,expected_trials=1. if k else 0.,
        classical=dict(nodes=c['nodes'],slots=c['slots'],geometry=c['geometry'],
            metadata=write_units*k+prep['classical_metadata']),
        quantum=dict(prepare_gate_total=prep['gate_total'],lookup_calls=2 if k else 0,
            geometry_calls=0,measurements=0,reset_rank_bits=0),
        query_table_writes=2*k,query_table_construction_units=write_units*k,
        reused_table_slot_capacity=2*n,active_table_words=2*k,
        rank_bits=bits,preparation=prep)
