import numpy as np

def add_readout_error(samples, p0=0.02, p1=0.03):
    """측정 결과에 readout error 추가"""
    noisy = []
    for s in samples:
        if s == 0 and np.random.rand() < p0:
            noisy.append(1)
        elif s == 1 and np.random.rand() < p1:
            noisy.append(0)
        else:
            noisy.append(s)
    return np.array(noisy)
