"""
Theoretical moments of the GA-SV family (theorems 2.1 to 2.3 and appendix B of the paper)
Used to compute the plug-in moments, the theoretical acf/ccf and the volatility surfaces
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfcx, gammaln, pbdv

# ---------------------------------------------------------------- GED errors
def abs_moment_G(p: int, nu: float) -> float:
    """
    Computing E(|epsilon|^p) for a standardized GED(nu) epsilon (nu > 1).
    """
    if nu <= 1.0:
        raise ValueError("nu must be strictly greater than 1.")
    if p == 0:
        return 1.0

    # 2^(1/nu) * lambda = sqrt(Gamma(1/nu) / Gamma(3/nu))
    log_scale = 0.5 * (gammaln(1.0 / nu) - gammaln(3.0 / nu))
    log_val = p * log_scale + gammaln((p + 1.0) / nu) - gammaln(1.0 / nu)
    return float(np.exp(log_val))

def expectation_term_G(
    s: float,
    nu: float,
    p: int = 0,
    signed: bool = False,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
    k_max: int = 150,
) -> float:
    """
    Computing E(|epsilon|^p * exp(s f(epsilon; theta))) (signed=False)
    or E(epsilon^p * exp(s f(epsilon; theta))) (signed=True) for a standardized GED(nu) epsilon (nu > 1).
    The expectation is written as a power series truncated at k_max.
    """
    if nu <= 1.0:
        raise ValueError("nu must be strictly greater than 1.")

    if s == 0.0:
        if signed and (p % 2 == 1):
            return 0.0
        return abs_moment_G(p, nu)

    log_scale = 0.5 * (gammaln(1.0 / nu) - gammaln(3.0 / nu))
    scale = np.exp(log_scale)

    # prefactor = exp(-s * gamma2 * E|epsilon|) * scale^p
    expected_abs_eps = np.exp(log_scale + gammaln(2.0 / nu) - gammaln(1.0 / nu))
    prefactor = np.exp(-s * gamma2 * expected_abs_eps + p * log_scale)

    k = np.arange(k_max + 1, dtype=np.float64)

    # log(Gamma((p + k + 1)/nu) / (2 * Gamma(1/nu) * k!))
    log_gamma_ratio = (
        gammaln((p + k + 1.0) / nu)
        - np.log(2.0)
        - gammaln(1.0 / nu)
        - gammaln(k + 1.0)
    )

    # (scale * s)^k * [(gamma1 + gamma2)^k + (+/-) exp(s * alpha) * (gamma2 - gamma1)^k]
    sign = (-1.0) ** p if signed else 1.0
    exp_s_alpha = np.exp(s * alpha)

    left = np.power(scale * s * (gamma1 + gamma2), k)
    right = sign * exp_s_alpha * np.power(scale * s * (gamma2 - gamma1), k)

    return float(prefactor * np.sum(np.exp(log_gamma_ratio) * (left + right)))

# ---------------------------------------------------------------- Normal errors
def abs_moment_N(p: int) -> float:
    """
    Computing E(|epsilon|^p) for epsilon ~ N(0, 1).
    """
    if p == 0:
        return 1.0
    if p == 1:
        return float(np.sqrt(2.0 / np.pi))
    if p == 2:
        return 1.0

    log_val = 0.5 * p * np.log(2.0) + gammaln(0.5 * (p + 1.0)) - 0.5 * np.log(np.pi)
    return float(np.exp(log_val))

def expectation_term_N(
    s: float,
    p: int = 0,
    signed: bool = False,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
) -> float:
    """
    Computing E(|epsilon|^p * exp(s f(epsilon; theta))) (signed=False)
    or E(epsilon^p * exp(s f(epsilon; theta))) (signed=True) for epsilon ~ N(0, 1).
    """
    if s == 0.0:
        if signed and (p % 2 == 1):
            return 0.0
        return abs_moment_N(p)

    C_bar = -s * (gamma1 - gamma2)
    D_bar = s * (gamma1 + gamma2)

    # prefactor = exp(-s * gamma2 * sqrt(2 / pi))
    prefactor = np.exp(-s * gamma2 * np.sqrt(2.0 / np.pi))
    exp_s_alpha = np.exp(s * alpha)
    sign = (-1.0) ** p if signed else 1.0

    # closed form with Phi / erfcx for p in {0, 1, 2} (appendix B.2)
    if p in {0, 1, 2}:
        expA_PhiC = 0.5 * erfcx(-C_bar / np.sqrt(2.0))
        expB_PhiD = 0.5 * erfcx(-D_bar / np.sqrt(2.0))

        if p == 0:
            return float(prefactor * (exp_s_alpha * expA_PhiC + expB_PhiD))

        if p == 1:
            left = sign * exp_s_alpha * (1.0 + np.sqrt(2.0 * np.pi) * C_bar * expA_PhiC)
            right = 1.0 + np.sqrt(2.0 * np.pi) * D_bar * expB_PhiD
            return float((prefactor / np.sqrt(2.0 * np.pi)) * (left + right))

        else:
            left = exp_s_alpha * (C_bar + np.sqrt(2.0 * np.pi) * (C_bar ** 2 + 1.0) * expA_PhiC)
            right = D_bar + np.sqrt(2.0 * np.pi) * (D_bar ** 2 + 1.0) * expB_PhiD
            return float((prefactor / np.sqrt(2.0 * np.pi)) * (left + right))

    # general form with the parabolic cylinder function D_{-p-1} for p > 2
    A_bar = 0.5 * C_bar ** 2
    B_bar = 0.5 * D_bar ** 2
    gamma_p1 = np.exp(gammaln(p + 1.0))

    left = sign * exp_s_alpha * np.exp(0.5 * A_bar) * pbdv(-p - 1, -C_bar)[0]
    right = np.exp(0.5 * B_bar) * pbdv(-p - 1, -D_bar)[0]
    return float((prefactor * gamma_p1 / np.sqrt(2.0 * np.pi)) * (left + right))

# ---------------------------------------------------------------- dispatch (Normal if nu is None or 2)
def abs_moment(p: int, nu: float | None = None) -> float:
    """
    Using the Normal formula if nu is None or nu == 2, and the GED formula otherwise.
    """
    if nu is None or nu == 2.0:
        return abs_moment_N(p)
    return abs_moment_G(p, nu)

def expectation_term(
    s: float,
    nu: float | None = None,
    p: int = 0,
    signed: bool = False,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
) -> float:
    """
    Using the Normal formula if nu is None or nu == 2, and the GED formula otherwise.
    """
    if nu is None or nu == 2.0:
        return expectation_term_N(s=s, p=p, signed=signed, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
    return expectation_term_G(s=s, nu=nu, p=p, signed=signed, alpha=alpha, gamma1=gamma1, gamma2=gamma2)

# ---------------------------------------------------------------- infinite products
def P(
    a: float,
    phi: float,
    nu: float | None = None,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
    tol: float = 1e-13,
    max_iter: int = 5000,
) -> float:
    """
    Approximating P(a, phi) = prod_{i=1}^inf E(exp(a * phi^(i-1) * f(epsilon))).
    """
    if abs(phi) >= 1.0:
        raise ValueError("Stationarity requires |phi| < 1.")
    if a == 0.0:
        return 1.0

    # ARSV (f = 0)
    if alpha == 0.0 and gamma1 == 0.0 and gamma2 == 0.0:
        return 1.0

    log_prod = 0.0
    s = float(a)

    for _ in range(max_iter):
        u_i = expectation_term(s=s, nu=nu, p=0, signed=False, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        log_u = np.log(u_i)
        log_prod += log_u

        # stop when the terms are smaller than the tolerance
        if abs(log_u) < tol:
            break
        s *= phi

    return float(np.exp(log_prod))

def Ts(
    a: float,
    phi: float,
    tau_max: int,
    nu: float | None = None,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
) -> np.ndarray:
    """
    Computing the vector [T_1(a, phi), ..., T_{tau_max}(a, phi)]
    with T_1 = 1 and T_tau = prod_{i=1}^{tau-1} E(exp(a * phi^(i-1) * f(epsilon))).
    """
    T = np.ones(tau_max, dtype=np.float64)
    if tau_max == 1 or a == 0.0:
        return T
    if alpha == 0.0 and gamma1 == 0.0 and gamma2 == 0.0:
        return T

    log_u = np.empty(tau_max - 1, dtype=np.float64)
    s = float(a)

    for idx in range(tau_max - 1):
        u_i = expectation_term(s=s, nu=nu, p=0, signed=False, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        log_u[idx] = np.log(u_i)
        s *= phi

    T[1:] = np.exp(np.cumsum(log_u))
    return T

# ---------------------------------------------------------------- theoretical acf and ccf (theorems 2.2 and 2.3)
def theoretical_acf(
    taus: np.ndarray,
    phi: float,
    sigma_eta: float,
    c: int = 2,
    nu: float | None = None,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
) -> np.ndarray:
    """
    Computing rho_c(tau) = corr(|y_t|^c, |y_{t+tau}|^c) for tau in taus.
    """
    taus = np.asarray(taus, dtype=int)

    expected_abs_eps_c = abs_moment(c, nu)
    expected_abs_eps_2c = abs_moment(2 * c, nu)

    P_half_c = P(0.5 * c, phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
    P_c = P(float(c), phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)

    base_term = (expected_abs_eps_c * P_half_c) ** 2
    denom = expected_abs_eps_2c * np.exp((c ** 2 * sigma_eta ** 2) / (4.0 * (1.0 - phi ** 2))) * P_c - base_term

    T = Ts(0.5 * c, phi, int(taus.max()), nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)

    rho = np.empty(len(taus), dtype=np.float64)
    for idx, tau in enumerate(taus):
        phi_tau = phi ** tau
        s_tau = 0.5 * c * (phi ** (tau - 1))

        expected_cross = expectation_term(s=s_tau, nu=nu, p=c, signed=False, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        P_tau = P(0.5 * c * (1.0 + phi_tau), phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        eta_term = np.exp((phi_tau * c ** 2 * sigma_eta ** 2) / (4.0 * (1.0 - phi ** 2)))

        num = expected_abs_eps_c * expected_cross * eta_term * P_tau * T[tau - 1] - base_term
        rho[idx] = num / denom

    return rho

def theoretical_ccf(
    taus: np.ndarray,
    phi: float,
    sigma_eta: float,
    c: int = 2,
    nu: float | None = None,
    alpha: float = 0.0,
    gamma1: float = 0.0,
    gamma2: float = 0.0,
) -> np.ndarray:
    """
    Computing rho_{c1}(tau) = corr(y_t, |y_{t+tau}|^c) for tau in taus.
    """
    taus = np.asarray(taus, dtype=int)

    expected_abs_eps_c = abs_moment(c, nu)
    expected_abs_eps_2c = abs_moment(2 * c, nu)

    P_half_c = P(0.5 * c, phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
    P_c = P(float(c), phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
    P_1 = P(1.0, phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)

    denom = np.sqrt(
        expected_abs_eps_2c * np.exp((c ** 2 * sigma_eta ** 2) / (4.0 * (1.0 - phi ** 2))) * P_c
        - (expected_abs_eps_c * P_half_c) ** 2
    )
    sqrt_P1 = np.sqrt(P_1)

    T = Ts(0.5 * c, phi, int(taus.max()), nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)

    rho_c1 = np.empty(len(taus), dtype=np.float64)
    for idx, tau in enumerate(taus):
        phi_tau = phi ** tau
        s_tau = 0.5 * c * (phi ** (tau - 1))

        expected_cross = expectation_term(s=s_tau, nu=nu, p=1, signed=True, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        P_tau = P(0.5 * (1.0 + c * phi_tau), phi, nu=nu, alpha=alpha, gamma1=gamma1, gamma2=gamma2)
        eta_term = np.exp(((2.0 * c * phi_tau - 1.0) * sigma_eta ** 2) / (8.0 * (1.0 - phi ** 2)))

        num = expected_abs_eps_c * eta_term * expected_cross * P_tau * (T[tau - 1] / sqrt_P1)
        rho_c1[idx] = num / denom

    return rho_c1

# ---------------------------------------------------------------- volatility response surface (equation 14)
def volatility_surface(eps_grid, eta_grid, alpha, gamma1, gamma2, nu=2.0, scale_factor=1.0):
    """
    Computing V_t = scale_factor * exp(f(epsilon_{t-1}) + eta_{t-1}) on a grid of (epsilon, eta).
    """
    E, H = np.meshgrid(eps_grid, eta_grid)
    e_abs = abs_moment(1, nu)
    f_val = alpha * (E < 0.0) + gamma1 * E + gamma2 * (np.abs(E) - e_abs)
    return E, H, scale_factor * np.exp(f_val + H)
