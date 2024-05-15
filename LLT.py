# -*- coding: utf-8 -*-



"""
Lifting Line Theory Application to a sample wing shape
Aircraft Aerodynamics 


"""

import numpy as np
import math
import matplotlib.pylab as plt

def lifting_line_theory(N, S, AR, taper, alpha_twist, i_w, a_2d, alpha_0):
    # Derived parameters
    b = math.sqrt(AR * S)  # Wing span (m)
    MAC = S / b  # Mean Aerodynamic Chord (m)
    Croot = (1.5 * (1 + taper) * MAC) / (1 + taper + taper ** 2)  # Root chord (m)

    # Discretization
    theta = np.linspace(math.pi / (2 * N), math.pi / 2, N, endpoint=True)
    alpha = np.linspace(i_w + alpha_twist, i_w, N)
    z = (b / 2) * np.cos(theta)
    c = Croot * (1 - (1 - taper) * np.cos(theta))  # Chord length distribution
    mu = c * a_2d / (4 * b)

    # Left Hand Side (LHS) of the equation
    LHS = mu * (np.array(alpha) - alpha_0) / 57.3  # Conversion from degrees to radians

    # Right Hand Side (RHS) of the equation
    RHS = []
    for i in range(1, 2 * N + 1, 2):
        RHS_iter = np.sin(i * theta) * (1 + (mu * i) / np.sin(theta))
        RHS.append(RHS_iter)

    RHS = np.asarray(RHS).T  # Transpose RHS to match dimensions for matrix operations
    inv_RHS = np.linalg.inv(RHS)  # Inverse of RHS matrix
    ans = np.matmul(inv_RHS, LHS)  # Solution for circulation distribution coefficients

    # Calculate lift coefficient distribution (CL)
    mynum = np.divide((4 * b), c)
    CL = sum((np.sin((2 * i - 1) * theta) * ans[i - 1] * mynum) for i in range(1, N + 1))

    # Plotting the lift distribution
    y_s = np.append([b / 2], z)  # Semi-span locations
    CL1 = np.append(0, CL)  # Append zero for plotting
    plt.plot(y_s, CL1, marker="o")
    plt.title("Lifting Line Theory\n Elliptical Lift distribution")
    plt.xlabel("Semi-span location (m)")
    plt.ylabel("Lift coefficient")
    plt.grid()
    plt.show()

    # Wing lift coefficient
    CL_wing = math.pi * AR * ans[0]  # Use this CL with cruise speed to calculate the accurate lift
    print(CL_wing, "CL_wing")

# Define different sample sets of parameters
samples = [
    (9, 5.49, 11.657, 0.25, 0, 1.0, 6.436, -1.2),
    (10, 6.0, 12.0, 0.3, 2, 1.5, 6.0, -1.0),
    (8, 4.5, 10.5, 0.35, -1, 0.5, 5.5, -1.5),
    (12, 5.8, 13.0, 0.2, 1, 1.2, 6.5, -1.1),
    (7, 5.0, 10.0, 0.4, 0.5, 1.0, 6.2, -1.3)
]

# Evaluate the function with different samples
for i, params in enumerate(samples, 1):
    print(f"Sample {i}: {params}")
    lifting_line_theory(*params)
    print("\n")
