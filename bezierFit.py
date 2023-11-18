import numpy as np
from scipy.linalg import qr, solve_triangular, lstsq
from scipy.special import comb
import torch


def bernstein_polynomial(i, n, t):
    """ Calculate the Bernstein polynomial value. """
    return comb(n, i) * ( t**i ) * (1 - t)**(n-i)

def fit_bezier_curves(samples, n_control_points=15):
    """ Fit bezier curves to the given samples. """
    batch_size, _, _ = samples.shape
    n_sample_points = 66  # Given as part of the problem

    # Output tensor
    fitted_curves = np.zeros((batch_size, n_control_points * 2, 2))

    for batch in range(batch_size):
        for dim in range(2):
            for curve in range(2):  # Upper and lower curve
                idx_start = curve * n_sample_points
                idx_end = idx_start + n_sample_points

                # Extracting the points for the current curve
                curve_points = samples[batch, idx_start:idx_end, dim]

                # Initialize control points
                control_points = np.zeros((n_control_points, 2))
                control_points[0, dim] = curve_points[0]
                control_points[-1, dim] = curve_points[-1]

                # Construct the coefficient matrix and b vector
                t_values = np.linspace(0, 1, n_sample_points)
                coeff_matrix = np.array([[bernstein_polynomial(j, n_control_points - 1, t) 
                                          for j in range(1, n_control_points - 1)] 
                                          for t in t_values])

                b_vector = curve_points - (control_points[0, dim] * bernstein_polynomial(0, n_control_points - 1, t_values) + 
                                           control_points[-1, dim] * bernstein_polynomial(n_control_points - 1, n_control_points - 1, t_values))

                # QR decomposition and solving the least squares problem
                Q, R = qr(coeff_matrix)
                y = np.dot(Q.T, b_vector)
                x, residuals, rank, s = lstsq(coeff_matrix, b_vector)

                # Set the control points
                control_points[1:-1, dim] = x

                # Update the fitted curves tensor
                fitted_curves[batch, curve * n_control_points:(curve + 1) * n_control_points, dim] = control_points[:, dim]
                tensor_cureves = torch.from_numpy(fitted_curves).float()


    return tensor_cureves

# Example usage
# samples = np.random.rand(batch_size, 66 * 2, 2)
# fitted_curves = fit_bezier_curves(samples)
