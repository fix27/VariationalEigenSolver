import cirq
from cirq.contrib.svg.svg import tdd_to_svg
import sympy


if __name__ == "__main__":
    qubits = cirq.LineQubit.range(10)
    x = sympy.symbols("x0:10")
    thetas = sympy.symbols("t0:10")
    circuit = cirq.Circuit()

    for i in range(10):
        circuit.append(cirq.rx(x[i])(qubits[i]))
        circuit.append(cirq.ry(thetas[i])(qubits[i]))

    circuit.append(cirq.measure_each(*qubits))

    with open("simple_VQC.svg", "w") as file:
        file.write(
            tdd_to_svg(
                circuit.to_text_diagram_drawer(), ref_boxheight=40, ref_boxwidth=160,
            )
        )

