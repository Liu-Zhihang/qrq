"""Unknown-K randomized AA policy and explicit matched-oracle resources.

K is used only by the offline expectation evaluator, never policy_windows().
This is not a hardware runtime or a complete multi-index quantum envelope.
"""
import functools,math
from .reversible_access import zero_reflection_circuit

def policy_windows(M):
    if not isinstance(M,int) or M<1:raise ValueError('positive candidate size required')
    cap=math.isqrt(M)+(math.isqrt(M)**2<M)
    m=1.
    for _ in range(300):
        w=min(cap,math.ceil(m))
        yield w
        if w==cap:return
        m=min(float(cap),m*1.2)
    raise RuntimeError('schedule bound')

def uniform_success(p,w):
    if p==0:return 0.
    if p==1:return 1.
    theta=math.asin(math.sqrt(p))
    value=.5-math.sin(4*w*theta)/(4*w*math.sin(2*theta))
    return min(1.,max(0.,value))

def draw_cost(w):
    if w==1:return 4.
    bits=(w-1).bit_length()
    return 4.+(bits+2)*2**bits/w

@functools.lru_cache(maxsize=100000)
def expectation(M,K):
    if not 0<K<=M:raise ValueError('paid witness requires 0<K<=M')
    p=K/M;survival=1.;attempts=iterations=control=0.
    windows=list(policy_windows(int(M)))
    for i,w in enumerate(windows):
        prob=uniform_success(p,w)
        visits=survival if i<len(windows)-1 else survival/prob
        attempts+=visits;iterations+=visits*(w-1)/2;control+=visits*draw_cost(w)
        survival*=1-prob
        if survival==0:break
    return dict(attempts=attempts,iterations=iterations,control=control,cap=windows[-1],stages=len(windows))

@functools.lru_cache(maxsize=None)
def reflection_gates(bits):return sum(zero_reflection_circuit(bits)['counts'].values())

def price(M,K,G,bits,P0,lookup=100,g_quantum=10000):
    e=expectation(int(M),int(K));a=e['attempts'];j=e['iterations'];marks=a+j
    vector=dict(prepare_calls=a+2*j,prepare_gates=(a+2*j)*G,
                predicate_calls=marks,lookup_calls=2*marks+2,
                reflection_gates=j*reflection_gates(bits),phase_aux_gates=4*j,
                measurements=a,reset_address_bits=bits*(a-1),classical_control=e['control'])
    cost=P0+vector['prepare_gates']+lookup*vector['lookup_calls']+g_quantum*marks+vector['reflection_gates']+vector['phase_aux_gates']+a+bits*(a-1)+e['control']
    return cost,e,vector
