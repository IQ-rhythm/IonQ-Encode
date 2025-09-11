import pennylane as qml
from pennylane import numpy as np
from readout_error import add_readout_error

dev = qml.device("default.mixed", wires=2, shots=512)

@qml.qnode(dev)
def ionq_like_circuit(theta, p1q=0.001, p2q=0.02):
    # 1-qubit gate with depolarizing
    qml.RX(theta, wires=0)
    qml.DepolarizingChannel(p1q, wires=0)

    # 2-qubit CNOT with depolarizing
    qml.CNOT(wires=[0,1])
    qml.DepolarizingChannel(p2q, wires=0)
    qml.DepolarizingChannel(p2q, wires=1)

    return qml.sample(qml.Z(0)), qml.sample(qml.Z(1))

if __name__ == "__main__":
    for shots in [256, 512, 1000]:
        dev.shots = shots
        s0, s1 = ionq_like_circuit(0.3)
        # add readout error
        s0_noisy = add_readout_error(s0, p0=0.01, p1=0.02)
        s1_noisy = add_readout_error(s1, p0=0.01, p1=0.02)
        print(f"shots={shots}, mean qubit0={np.mean(s0_noisy)}, qubit1={np.mean(s1_noisy)}")
