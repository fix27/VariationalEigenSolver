import os
import sys
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
from cirq.contrib.svg.svg import tdd_to_svg
from scipy import sparse
from scipy.sparse import linalg
from tqdm import tqdm


def get_ising_operator(
    qubits: List[cirq.GridQubit], j: float, h: float
) -> cirq.PauliSum:
    """Create an Ising (TFI) operator.

    Args:
        qubits (List[cirq.GridQubit]): qubits.
        j (float): j constant.
        h (float): h constant,

    Returns:
        cirq.PauliSum: TFI operator
    """
    op = h * cirq.X(qubits[-1])

    for i, _ in enumerate(qubits[:-1]):
        op -= j * cirq.Z(qubits[i]) * cirq.Z(qubits[i + 1])
        op += h * cirq.X(qubits[i])

    return op


def quantum_solve(
    dim: int, h: float, j: float = 1, epochs: int = 350, lr=1e-2
) -> float:
    """Variational Quantum Eigensolver.

    Args:
        dim (int): dimension of problem.
        h (float): TFI parameter.
        j (float, optional): TFI parameter. Defaults to 1.
        epochs (int, optional): number of epochs. Defaults to 350.
        lr ([type], optional): learning rate. Defaults to 1e-2.

    Returns:
        float: estimation of minimal eigenvalue
    """
    qubits = cirq.GridQubit.rect(dim, 1)
    params_x = sympy.symbols(f"x0:{dim}")
    params_y = sympy.symbols(f"y0:{dim}")
    params_z = sympy.symbols(f"z0:{dim}")
    cirquit = cirq.Circuit()

    for i in range(dim):
        cirquit.append(cirq.rx(params_x[i])(qubits[i]))
        cirquit.append(cirq.ry(params_y[i])(qubits[i]))
        cirquit.append(cirq.rz(params_z[i])(qubits[i]))

    with open("circuit.svg", "w") as file:
        svg_str = tdd_to_svg(
            cirquit.to_text_diagram_drawer(transpose=False),
            ref_boxheight=40,
            ref_boxwidth=120,
        )
        file.write(svg_str)

    op = get_ising_operator(qubits, j, h)
    model = tfq.layers.SampledExpectation()

    thetas = tf.Variable(np.random.random((1, 3 * dim)), dtype=tf.float32)

    exact_sol = linalg.eigs(
        sparse.csc_matrix(op.matrix()), k=1, which="SR", return_eigenvectors=False
    )[0]
    print(f"Exact solution: {np.real(exact_sol):.4f}")

    start_val = model(
        cirquit,
        symbol_names=params_x + params_y + params_z,
        symbol_values=thetas,
        operators=op,
        repetitions=5000,
    )

    print(f"Initialized energy: {start_val.numpy()[0][0]:.4f}")

    log_writer = tf.summary.create_file_writer("train")

    for epoch in tqdm(range(epochs)):
        with tf.GradientTape() as gt:
            out = model(
                cirquit,
                symbol_names=params_x + params_y + params_z,
                symbol_values=thetas,
                operators=op,
                repetitions=5000,
            )

        grad = gt.gradient(out, thetas)
        thetas.assign_sub(lr * grad)

        with log_writer.as_default():
            tf.summary.scalar("Eigen Val", out[0, 0], step=epoch)
            tf.summary.histogram("Gradients", grad, step=epoch)

    solution = model(
        cirquit,
        symbol_names=params_x + params_y + params_z,
        symbol_values=thetas,
        operators=op,
        repetitions=10000,
    )

    print(f"VQE solution: {solution.numpy()[0][0]:.4f}")

    return solution.numpy()[0][0]


if __name__ == "__main__":
    n = int(sys.argv[1])
    h = float(sys.argv[2])
    quantum_solve(n, h)
