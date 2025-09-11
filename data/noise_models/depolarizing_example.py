#단일 큐빗에 qml.DepolarizingChannel(p)을 직접 삽입

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.mixed", wires=1, shots=512)

@qml.qnode(dev)
def noisy_circuit(theta, p=0.01):
    qml.RX(theta, wires=0)
    qml.DepolarizingChannel(p, wires=0)
    return qml.sample(qml.Z(0))

if __name__ == "__main__":
    for shots in [256, 512, 1000]:
        dev.shots = shots
        samples = noisy_circuit(0.5, p=0.05)
        print(f"shots={shots}, mean={np.mean(samples)}")
