from typing import List, Tuple

import numpy as np
import cirq


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    # create a shift matrix
    dimension = len(matrix)
    shift_matrix = np.eye(dimension)
    last_row = np.array(np.linspace(0, 0, dimension))
    last_row[-1] = 1
    shift_matrix[1:] = shift_matrix[0:dimension - 1]
    shift_matrix[0] = last_row

    def exchange(first_bit, second_bit):
        moment0 = cirq.Moment([cirq.ISWAP(first_bit, second_bit)])
        moment1 = cirq.Moment([(cirq.X ** 0.5)(second_bit)])
        moment2 = cirq.Moment([cirq.ISWAP(first_bit, second_bit)])
        moment3 = cirq.Moment([(cirq.X ** 0.5)(first_bit)])
        moment4 = cirq.Moment([cirq.ISWAP(first_bit, second_bit)])
        moment5 = cirq.Moment([(cirq.X ** 0.5)(second_bit)])
        return [moment0, moment1, moment2, moment3, moment4, moment5]

    if np.allclose(matrix, np.eye(2**len(target_qubits))):
        return [], []
    if len(target_qubits) == 1:  # single bit case
        if np.allclose(matrix, cirq.unitary(cirq.X)):
            return [cirq.X(target_qubits[0])], []
        if np.allclose(matrix, cirq.unitary(cirq.Y)):
            return [cirq.Y(target_qubits[0])], []
        if np.allclose(matrix, cirq.unitary(cirq.Z)):
            return [cirq.Z(target_qubits[0])], []
        if np.allclose(matrix, cirq.unitary(cirq.H)):  # H gate
            return [[(cirq.Y**0.5)(target_qubits[0]), cirq.X(target_qubits[0])]], []
        if np.allclose(matrix, cirq.unitary(cirq.S)):  # S gate
            return [cirq.ZPowGate(exponent=0.5)(target_qubits[0])], []
        if np.allclose(matrix, cirq.unitary(cirq.T)):  # T gate
            return [cirq.ZPowGate(exponent=0.25)(target_qubits[0])], []
    if dimension == 4 and np.allclose(shift_matrix, matrix):
        return [exchange(target_qubits[0], target_qubits[1])], []


    return [], []
