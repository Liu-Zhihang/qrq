"""Versioned first-witness valid-rank reference and matched-output resources.

No hardware timings; static coherent record access and clean geometry are
explicitly priced oracles in city-scale comparisons.
"""
from collections import Counter
from functools import lru_cache
import math
import numpy as np
from .cities_reversible_access import mcx, gate_counts, apply_state


def merge_ranges(ranges):
    merged=[]; previous=-1
    for lo,hi in ranges:
        lo,hi=int(lo),int(hi)
        if not 0<=lo<hi or lo<previous:
            raise ValueError('ranges must be sorted, disjoint and positive')
        previous=hi
        if merged and merged[-1][1]==lo: merged[-1][1]=hi
        else: merged.append([lo,hi])
    return merged


def lower_operation(bits, prefix, depth, target, kind, angle=0.):
    controls=list(range(bits-1,bits-depth-1,-1))
    zeros=[w for j,w in enumerate(controls) if not (prefix>>(depth-1-j))&1]
    ancillas=list(range(bits,bits+max(0,bits-1)))
    gates=[('X',w) for w in zeros]
    if kind=='H':
        assert not controls
        gates.append(('H',target))
    elif kind=='X': mcx(gates,controls,target,ancillas)
    elif kind=='RY':
        if not controls: gates.append(('RY',target,angle))
        else:
            before=[]
            selector=controls[0] if len(controls)==1 else ancillas[-1]
            if len(controls)>1: mcx(before,controls,selector,ancillas[:-1])
            gates.extend(before)
            gates.extend((('RY',target,angle/2),('CX',selector,target),
                          ('RY',target,-angle/2),('CX',selector,target)))
            gates.extend(reversed(before))
    else: raise ValueError(kind)
    gates.extend(('X',w) for w in reversed(zeros))
    return gates


@lru_cache(maxsize=None)
def operation_counts(bits,depth,zero_count,kind):
    prefix=(1<<(depth-zero_count))-1
    return gate_counts(lower_operation(bits,prefix,depth,bits-depth-1,kind,.7))


def compile_intervals(ranges,bits,with_program=False):
    """O(boundary prefixes) compiler: whole binary subtrees never enumerate IDs."""
    raw_count=len(ranges);merged=merge_ranges(ranges)
    if not bits>=1 or (merged and merged[-1][1]>1<<bits): raise ValueError('address bounds')
    starts=[r[0] for r in merged];cum=[0]
    for lo,hi in merged: cum.append(cum[-1]+hi-lo)
    comparisons=0; angles=0; counts=Counter(gate_counts([]));program=[];visited=0
    def mass_before(x):
        nonlocal comparisons
        lo,hi=0,len(starts)
        while lo<hi:
            mid=(lo+hi)//2;comparisons+=1
            if starts[mid]<=x:lo=mid+1
            else:hi=mid
        index=lo-1
        if index<0:return 0
        return cum[index]+min(x-starts[index],merged[index][1]-starts[index])
    def emit(prefix,depth,target,kind,angle):
        counts.update(operation_counts(bits,depth,depth-prefix.bit_count(),kind))
        if with_program:program.append((prefix,depth,target,kind,angle))
    def visit(prefix,depth,weight,base):
        nonlocal angles,visited
        visited+=1
        if depth==bits or not weight:return
        remaining=bits-depth;target=remaining-1
        if weight==1<<remaining:
            for t in range(target,-1,-1):emit(prefix,depth,t,'H' if depth==0 else 'RY',math.pi/2)
            return
        midpoint=(prefix*2+1)<<target
        left=mass_before(midpoint)-base;right=weight-left
        if right:
            if not left:emit(prefix,depth,target,'X',math.pi)
            else:
                angles+=1;emit(prefix,depth,target,'RY',2*math.atan2(math.sqrt(right),math.sqrt(left)))
        if left:visit(prefix*2,depth+1,left,base)
        if right:visit(prefix*2+1,depth+1,right,base+left)
    if cum[-1]:visit(0,0,cum[-1],0)
    return dict(ranges=merged,domain=cum[-1],raw_ranges=raw_count,merged_ranges=len(merged),
        address_bits=bits,qubits=bits+max(0,bits-1),counts=dict(counts),
        gate_total=sum(counts.values()),lookup_comparisons=comparisons,angle_evaluations=angles,
        # Two descriptor words, one validation and one merging visit per input range.
        classical_metadata=4*raw_count+comparisons+angles,
        visited_prefixes=visited,program=program if with_program else None)


def simulate_preparation(compiled):
    if compiled['program'] is None:raise ValueError('program required')
    zero=np.zeros(1<<compiled['qubits'],complex);zero[0]=1
    gates=[]
    for p,d,t,k,a in compiled['program']:gates.extend(lower_operation(compiled['address_bits'],p,d,t,k,a))
    assert dict(gate_counts(gates))==compiled['counts']
    state=apply_state(gates,zero)
    back=apply_state(gates,state,inverse=True)
    assert np.linalg.norm(back-zero)<1e-10
    capacity=1<<compiled['address_bits']
    assert np.linalg.norm(state[capacity:])<1e-10
    return state[:capacity],gates


def route_resources(plan,compiled,k,route,bits):
    quantum=route=='QRQ_FW'
    domain=compiled['domain']
    if quantum and (bool(domain)!=bool(k)):
        raise AssertionError('witness and nonemptiness disagree')
    a=domain/k if quantum and k else (1. if k else 0.)
    classical=dict(nodes=int(plan['node_reads']),slots=int(plan['slots']),
        geometry=int(plan['geometry']),metadata=compiled['classical_metadata'])
    resources=dict(prepare_gate_total=a*compiled['gate_total'],
        lookup_calls=(2*a+2 if quantum else 2.) if k else 0.,
        geometry_calls=a if quantum else 0.,measurements=a if quantum else 0.,
        reset_rank_bits=bits*(a-1) if quantum and k else 0.)
    return dict(route=route,domain=domain,expected_trials=a,
        classical=classical,quantum=resources,
        preparation={key:compiled[key] for key in ('raw_ranges','merged_ranges','gate_total',
            'counts','lookup_comparisons','angle_evaluations','visited_prefixes','classical_metadata')},
        shared_initial_rank_bits=bits,record_and_predicate_workspace='shared_clean_oracle_contract_not_city_gate_synthesized')


def price(row,g_classical=1,lookup=1,g_quantum=1,lam=1,repeats=1):
    c=row['classical'];q=row['quantum']
    classical=c['nodes']+c['slots']+g_classical*c['geometry']+c['metadata']
    quantum=q['prepare_gate_total']+lookup*q['lookup_calls']+g_quantum*q['geometry_calls']+q['measurements']+q['reset_rank_bits']
    return classical+lam*repeats*quantum
