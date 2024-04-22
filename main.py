import numpy as np


def STLS_REGRESSION(X_dot, theta_T, epsilon, max_iterations):
    # Initialize sparse coefficient matrix
    X_hat = np.linalg.pinv(theta_T).dot(X_dot)  # Initial least squares guess

    # Start iteration
    for iteration in range(max_iterations):
        # Find small entries
        is_small = np.abs(X_hat) < epsilon  # 找到小的元素

        # Thresholding
        X_hat[is_small] = 0    #赋值为零，6

        # Regress onto big terms
        for ii in range(X_dot.shape[1]):
            is_big = ~is_small[:, ii]
            if np.any(is_big):
                X_hat[is_big, ii] = np.linalg.pinv(theta_T[:, is_big]).dot(X_dot[:, ii])

    return X_hat


# Example usage
# X_dot: Time derivative
# theta_T: Library of candidate functions
# epsilon: Thresholding parameter
# max_iterations: Maximum number of iterations
X_dot = np.random.rand(3, 10)  # Example random time derivative matrix
theta_T = np.random.rand(10, 84)  # Example random candidate function library
epsilon = 0.1  # Example thresholding parameter
max_iterations = 100  # Example maximum number of iterations

# Call the function
sparse_coefficients = STLS_REGRESSION(X_dot, theta_T, epsilon, max_iterations)
print(sparse_coefficients)

print("hello world")