"""
Theoretical properties of the GA-SV models (sections 2 and 3 of the paper)
Reproducing: figure 1 (rho_2(1) and rho_21(1) against alpha and gamma1), figure 2 (theoretical acf and ccf for
different GED parameters nu) and figure 3 (volatility response surfaces)
Images are saved in results/images
"""
import sys
sys.path.append(".")
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.theory import theoretical_acf, theoretical_ccf, volatility_surface

# parameters used in the three figures
phi = 0.98
sigma_eta = np.sqrt(0.05)

# TGA-SV, A-ARSV, E-SV and RT-SV (figures 2 and 3)
models = [
    ("TGA-SV", dict(alpha=0.07, gamma1=-0.08, gamma2=0.1)),
    ("A-ARSV", dict(alpha=0.0, gamma1=-0.08, gamma2=0.0)),
    ("E-SV",   dict(alpha=0.0, gamma1=-0.08, gamma2=0.1)),
    ("RT-SV",  dict(alpha=0.07, gamma1=0.0, gamma2=0.0)),
]

nu_styles = [
    (1.5, "red", "-",   r"$\nu = 1.5$"),
    (1.7, "green", "--", r"$\nu = 1.7$"),
    (2.0, "blue", ":",  r"$\nu = 2.0$"),
    (2.5, "black", "-.", r"$\nu = 2.5$"),
]

def figure1():
    """
    Figure 1: first-order acf of squared returns rho_2(1) (left) and cross-correlation rho_21(1) (right) for Gaussian
    TGA-SV models as a function of (alpha, gamma1), with gamma2 in {0, 0.1}.
    Output: results/images/figure1_rho2_rho21_vs_alpha_gamma1.png
    """
    g1_grid = np.linspace(-0.25, 0.0, 21)
    a_grid = np.linspace(0.0, 1.0, 21)
    A, G1 = np.meshgrid(a_grid, g1_grid)

    fig = plt.figure(figsize=(13, 5.5), dpi=150)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for gamma2, edgecolor in [(0.0, "navy"), (0.1, "teal")]:
        Z_acf = np.empty_like(G1)
        Z_ccf = np.empty_like(G1)
        for i in range(G1.shape[0]):
            for j in range(G1.shape[1]):
                Z_acf[i, j] = theoretical_acf(np.array([1]), phi, sigma_eta, c=2, nu=2.0,
                                              alpha=A[i, j], gamma1=G1[i, j], gamma2=gamma2)[0]
                Z_ccf[i, j] = theoretical_ccf(np.array([1]), phi, sigma_eta, c=2, nu=2.0,
                                              alpha=A[i, j], gamma1=G1[i, j], gamma2=gamma2)[0]
        ax1.plot_wireframe(A, G1, Z_acf, color=edgecolor, linewidth=0.7)
        ax2.plot_wireframe(A, G1, Z_ccf, color=edgecolor, linewidth=0.7)

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.25, 0.0)
    ax1.set_zlim(0.0, 0.4)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\gamma_1$")
    ax1.set_zlabel(r"$\rho_2(1)$")
    ax1.view_init(elev=22, azim=-125)

    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.25, 0.0)
    ax2.set_zlim(-0.25, 0.0)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel(r"$\gamma_1$")
    ax2.set_zlabel(r"$\rho_{21}(1)$")
    ax2.view_init(elev=22, azim=-125)

    plt.tight_layout()
    plt.savefig("results/images/figure1_rho2_rho21_vs_alpha_gamma1.png")
    plt.close()

def figure2():
    """
    Figure 2: theoretical acf of squared returns (top) and ccf between returns and future squared returns (bottom),
    tau = 1, ..., 40, for TGA-SV, A-ARSV, E-SV and RT-SV with GED errors and nu in {1.5, 1.7, 2.0, 2.5}.
    Output: results/images/figure2_theoretical_acf_ccf_by_nu.png
    """
    taus = np.arange(1, 41)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), dpi=150)

    for col, (name, params) in enumerate(models):
        for nu, color, ls, label in nu_styles:
            rho2 = theoretical_acf(taus, phi=phi, sigma_eta=sigma_eta, c=2, nu=nu, **params)
            rho21 = theoretical_ccf(taus, phi=phi, sigma_eta=sigma_eta, c=2, nu=nu, **params)
            axes[0, col].plot(taus, rho2, color=color, linestyle=ls, linewidth=1.3, label=label)
            axes[1, col].plot(taus, rho21, color=color, linestyle=ls, linewidth=1.3, label=label)

        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_ylim(0.0, 0.35)
        axes[1, col].set_ylim(-0.06, 0.0)
        for row in range(2):
            axes[row, col].set_xlim(0, 40)
            axes[row, col].set_xticks([0, 20, 40])
            axes[row, col].set_xlabel(r"$\tau$")

    axes[0, 0].set_ylabel(r"$\rho_2(\tau)$")
    axes[1, 0].set_ylabel(r"$\rho_{21}(\tau)$")
    axes[0, 0].legend(fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig("results/images/figure2_theoretical_acf_ccf_by_nu.png")
    plt.close()

def figure3():
    """
    Figure 3: theoretical volatility response surfaces V_t as a function of (epsilon_{t-1}, eta_{t-1}) for TGA-SV,
    A-ARSV and E-SV with Gaussian errors (normalized so that the scale factor is 1).
    Output: results/images/figure3_volatility_response_surfaces.png
    """
    eps_grid = np.linspace(-2.0, 2.0, 60)
    eta_grid = np.linspace(-2.0, 2.0, 60)

    fig = plt.figure(figsize=(7, 13), dpi=130)
    for idx, (title, params) in enumerate(models[:3], 1):
        ax = fig.add_subplot(3, 1, idx, projection="3d")
        E, H, V = volatility_surface(eps_grid, eta_grid, nu=2.0, scale_factor=1.0, **params)
        ax.plot_surface(E, H, V, cmap="jet", edgecolor="k", linewidth=0.15)
        ax.set_xlabel(r"$\epsilon_{t-1}$")
        ax.set_ylabel(r"$\eta_{t-1}$")
        ax.set_zlabel(r"$V_t$")
        ax.set_title(title)
        ax.view_init(elev=25, azim=-125)

    plt.tight_layout()
    plt.savefig("results/images/figure3_volatility_response_surfaces.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("results/images", exist_ok=True)

    figure1()
    figure2()
    figure3()
