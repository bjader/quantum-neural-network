import logging

import numpy as np
from qiskit import assemble, transpile


def calculate_gradient_list(qnn, parameter_list, method='parameter shift', shots=100, eps=None):
    parameter_list = np.array(parameter_list, dtype=float)

    if method == 'parameter shift':
        r = 0.5  # for Farhi ansatz, e0 = -1, e1 = +1, a = 1 => r = 0.5 (Using notation in arXiv:1905.13311)

        qc_plus_list, qc_minus_list = get_parameter_shift_circuits(qnn, parameter_list, r)

        expectation_minus, expectation_plus = evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots)

        gradient_list = r * (expectation_plus - expectation_minus)

    elif method == 'finite difference':
        qc_plus_list, qc_minus_list = get_finite_difference_circuits(qnn, parameter_list, eps)

        expectation_minus, expectation_plus = evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots)

        gradient_list = (expectation_plus - expectation_minus) / (2 * eps)

    else:
        raise ValueError("Invalid gradient method")

    gradient_list = gradient_list.reshape([len(parameter_list), -1])
    return gradient_list


def evaluate_gradient_jobs(qc_minus_list, qc_plus_list, qnn, shots):
    qc_minus_list = [transpile(circ, basis_gates=['cx', 'u1', 'u2', 'u3']) for circ in qc_minus_list]
    qc_plus_list = [transpile(circ, basis_gates=['cx', 'u1', 'u2', 'u3']) for circ in qc_plus_list]
    job = assemble(qc_minus_list + qc_plus_list, backend=qnn.backend, shots=shots)
    results = qnn.backend.run(job).result()
    expectation_plus = []
    expectation_minus = []
    num_params = len(qc_plus_list)
    for i in range(num_params):
        expectation_minus.append(qnn.get_vector_from_results(results, i))
        expectation_plus.append(qnn.get_vector_from_results(results, num_params + i))
        logging.debug("Gradient calculated for {} out of {} parameters".format(i, num_params))
    return np.array(expectation_minus), np.array(expectation_plus)


def get_parameter_shift_circuits(qnn, parameter_list, r):
    qc_plus_list, qc_minus_list = [], []
    for i in range(len(parameter_list)):
        shifted_params_plus = np.copy(parameter_list)
        shifted_params_plus[i] = shifted_params_plus[i] + np.pi / (4 * r)
        shifted_params_minus = np.copy(parameter_list)
        shifted_params_minus[i] = shifted_params_minus[i] - np.pi / (4 * r)

        qc_i_plus = qnn.bind_circuit(shifted_params_plus)
        qc_i_minus = qnn.bind_circuit(shifted_params_minus)
        qc_plus_list.append(qc_i_plus)
        qc_minus_list.append(qc_i_minus)

    return qc_plus_list, qc_minus_list


def get_finite_difference_circuits(qnn, parameter_list, eps):
    if type(eps) == float:
        qc_plus_list, qc_minus_list = [], []
        for i in range(len(parameter_list)):
            shifted_params_plus = np.copy(parameter_list)
            shifted_params_plus[i] = shifted_params_plus[i] + eps
            shifted_params_minus = np.copy(parameter_list)
            shifted_params_minus[i] = shifted_params_minus[i] - eps

            qc_i_plus = qnn.bind_circuit(shifted_params_plus)
            qc_i_minus = qnn.bind_circuit(shifted_params_minus)
            qc_plus_list.append(qc_i_plus)
            qc_minus_list.append(qc_i_minus)

        return qc_plus_list, qc_minus_list
    else:
        raise ValueError("eps for finite difference scheme must be float")
