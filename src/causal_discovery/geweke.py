import numpy as np
from statsmodels.tsa.api import VAR
from scipy.linalg import inv


def geweke_spectral_causality(process: np.ndarray, maxlags: int = 1, n_freqs: int = 256) -> np.ndarray:

    n_vars, _ = process.shape
    causality_matrix = np.zeros((n_vars, n_vars))

    for caused in range(n_vars):
        for causing in range(n_vars):
            if caused == causing:
                continue

            try:
                data_pair = process[[causing, caused]].T
                model = VAR(data_pair)
                results = model.fit(maxlags=maxlags)

                A_hat = results.coefs[0]
                Sigma_u = results.sigma_u

                freqs = np.linspace(0, 0.5, n_freqs)
                I = np.eye(2)
                G_vals = []

                for f in freqs:
                    z = np.exp(-2j * np.pi * f)
                    A_z = I - A_hat * z
                    H_f = inv(A_z)
                    S_f = H_f @ Sigma_u @ H_f.conj().T

                    S22 = S_f[1, 1]
                    S21 = S_f[1, 0]
                    S11 = S_f[0, 0]
                    S22_cond = S22 - (np.abs(S21) ** 2) / S11
                    G_f = np.log(S22 / S22_cond).real
                    G_vals.append(G_f)

                causality_matrix[causing, caused] = np.mean(G_vals)

            except Exception as e:
                print(f"Error Geweke {causing} → {caused}: {e}")
                continue


    np.fill_diagonal(causality_matrix, 0)
    return (causality_matrix != 0).astype(int)