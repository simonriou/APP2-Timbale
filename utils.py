from scipy.special import jv, jn_zeros
from scipy.optimize import brentq
import numpy as np


DEFAULT_VISUALISED_MODES = (
    (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
)


def get_nm(fmax, a, c):
    """
    Calcule itérativement les k_nm jusqu'à ce que f_nm > f_max.
    Retourne n_max, m_max
    """
    n_max = 0
    m_max = 0
    while True:
        k_nm = jn_zeros(n_max + 1, m_max + 1)[-1] / a
        f_nm = k_nm * c / (2 * np.pi)
        if f_nm > fmax:
            break
        if m_max < n_max:
            m_max += 1
        else:
            n_max += 1
            m_max = 0
    return n_max, m_max

def calculer_kn_m0(a, T, V0, rho0, c0, N_modes):
    alpha = (rho0 * (c0**2) * np.pi * (a**2)) / (V0 * T)

    def equation(k):
        return (k**2) * jv(0, k * a) + alpha * jv(2, k * a)

    racines = []
    step = np.pi / (4.0 * a)
    k_current = 1e-6

    while len(racines) < N_modes:
        k_next = k_current + step
        if equation(k_current) * equation(k_next) <= 0:
            try:
                racine = brentq(equation, k_current, k_next)
                if abs(jv(0, racine * a)) > 1e-12:
                    racines.append(racine)
            except ValueError:
                pass
        k_current = k_next
        
    return racines

def calculer_distances_acoustiques(R_obs, phi_obs, theta_obs, a, N_r=50, N_theta=100, c0=343.0, fs=44100):
    """
    Calcule les distances selon les approximations de Fraunhofer et Fresnel.
    """
    r_vec = np.linspace(1e-5, a, N_r)
    theta_vec = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)
    grille_r, grille_theta = np.meshgrid(r_vec, theta_vec, indexing='ij')
    
    # Terme m.r' commun aux deux approximations (projection axiale)
    projection = grille_r * np.sin(phi_obs) * np.cos(theta_obs - grille_theta)
    
    # 1. Approximation de Fraunhofer (2 termes)
    distances_fraunhofer = R_obs - projection
    
    # 2. Approximation de Fresnel (3 termes avec le terme quadratique)
    # L'hypothèse de Fresnel suppose R_obs (x) pas trop éloigné de l'axe
    distances_fresnel = R_obs - projection + (grille_r**2) / (2 * R_obs)
    
    dr = a / N_r
    dtheta = 2 * np.pi / N_theta
    dS_i = grille_r * dr * dtheta
    
    return distances_fraunhofer, distances_fresnel, dS_i, grille_r, grille_theta

def plot_modes(a, modes=DEFAULT_VISUALISED_MODES, grid_size=300, save_path=None, show=False, cuve=False, k_n0=None):
    """
    Plot the spatial shapes of circular membrane modes.
    """
    import matplotlib.pyplot as plt
    from synthese import phi

    if a <= 0:
        raise ValueError("a doit être strictement positif")
    if grid_size < 2:
        raise ValueError("grid_size doit être supérieur ou égal à 2")

    validated_modes = []
    for mode in modes:
        if len(mode) != 2:
            raise ValueError("Chaque mode doit être un couple (m, n)")
        m, n = mode
        validated_modes.append((int(m), int(n)))

    coordinates = np.linspace(-a, a, grid_size)
    x, y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(x, y)
    angle = np.arctan2(y, x)
    inside_membrane = radius <= a

    figure, axes = plt.subplots(
        3, 3, figsize=(12, 12), constrained_layout=True, squeeze=False,
    )
    axes_flat = axes.ravel()

    for axis, (m, n) in zip(axes_flat, validated_modes):
        # 1. Select the correct wavenumber based on the coupling state
        if cuve and m == 0 and k_n0 is not None:
            k_nm = k_n0[n-1]
        else:
            k_nm = jn_zeros(m, n)[-1] / a
            
        # 2. Pass the 'cuve' variable instead of False
        mode_shape = phi(k_nm, m, radius, angle, 0.0, a, cuve)
        
        mode_shape = np.ma.masked_where(~inside_membrane, mode_shape)
        axis.pcolormesh(
            x, y, mode_shape, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1,
        )
        axis.set_title(f"Mode (m={m}, n={n})")
        axis.set_aspect("equal")
        axis.set_xlim(-a, a)
        axis.set_ylim(-a, a)
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    for axis in axes_flat[len(validated_modes):]:
        axis.set_visible(False)

    figure.suptitle("Déformées modales de la membrane")
    if save_path is not None:
        figure.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return figure, axes