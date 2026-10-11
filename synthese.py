import json
from pathlib import Path

from scipy.special import jv, jn_zeros
import numpy as np
from scipy.io.wavfile import write

from utils import plot_modes, calculer_kn_m0, calculer_distances_acoustiques

def phi(k, m, r, theta, theta0, a, cuve):
    """
    Calcule la valeur de la déformée modale liée au mode (n, m) (n est déjà absorbé dans k) à la position (r, theta).
    theta0 est l'angle d'excitation.
    """
    if k == 0:
        return 0.0 # La déformée modale pour k=0 est nulle
    if cuve:
        if m == 0:
            return jv(0, k * r) - jv(0, k * a)
        else:
            return jv(m, k * r) * np.cos(m * (theta - theta0))
    else:
        return jv(m, k * r) * np.cos(m * (theta - theta0))

def synthese_membrane(a, c, beta, eta, V0, rhoS, rho0, c0, r_0, theta_0, F_0, r_lst, theta_lst, phi_lst, duree, fs, N_modes, M_modes, cuve, rayonnement, lst_membrane):
    """
    Synthèse modale de la membrane sous les hypothèses suivantes:
    - Circulaire, de rayon a, tendue uniformément
    - Couplée à une cuve (raideur uniforme), pas de rayonnement
    - Excitation ponctuelle en r, theta et t
    - Pertes: frottements visqueux (air) et viscoélasticité (tension de la membrane)
    """

    if N_modes < 1 or M_modes < 0:
        raise ValueError("N_modes doit être >= 1 et M_modes >= 0")

    if cuve:
        C0 = (1.4 * 10e5 / V0 * rhoS)
        k_n0 = calculer_kn_m0(a, c**2 * rhoS, V0, rho0, c0, N_modes)

    t = np.linspace(0, duree, int(duree * fs))

    signal_total = np.zeros_like(t)

    if rayonnement:
        distances_fraunhofer, distances_fresnel, dSi, grille_r, grille_theta = calculer_distances_acoustiques(r_lst, phi_lst, theta_lst, a, c0=c0, fs=fs)

    for n in range(1, N_modes + 1):
        for m in range(0, M_modes + 1):
            k_nm = jn_zeros(m, n)[-1] / a
            if cuve:
                if m == 0:
                    k_nm = k_n0[n-1]
                
            omega_nm = k_nm * c
            delta_nm = beta + 0.5 * eta * k_nm**2 # décroissance exp. = 0 si pas d'amortissement
            omega_d = np.sqrt(omega_nm**2 - delta_nm**2) # fréquence propre amortie = omega_nm si pas d'amortissement

            if cuve and m == 0:
                # Intégration numérique stricte car la déformée est modifiée par la cuve
                r_eval = np.linspace(0, a, 200)
                phi_eval = np.array([phi(k_nm, 0, r_val, 0, 0, a, cuve) for r_val in r_eval])
                    
                # Intégration polaire: 2*pi*r dr
                N_nm = np.trapezoid((phi_eval**2) * 2 * np.pi * r_eval, r_eval)
            else:
                # Formule analytique pour modes libres (ou asymétriques)
                facteur_angle = 2 * np.pi if m == 0 else np.pi
                N_nm = facteur_angle * (a**2 / 2.0) * (jv(m + 1, k_nm * a)**2)
            M_tot = rhoS * N_nm
            R_tot = 2 * beta * rhoS * N_nm

            if rayonnement:
                K_tot = rhoS * N_nm * omega_nm**2
                S = np.pi * a**2 # Correction de la surface du piston

                M_nm = rhoS * N_nm
                K_tot = rhoS * N_nm * omega_nm**2

                R_nm = 0

                if m == 0:
                    # Intégration numérique stricte car la déformée est modifiée par la cuve
                    r_eval = np.linspace(0, a, 200)
                    phi_eval = np.array([phi(k_nm, 0, r_val, 0, 0, a, cuve) for r_val in r_eval])
                    # Intégration polaire: 2*pi*r dr
                    moyenne = np.trapezoid(phi_eval * 2 * np.pi * r_eval, r_eval) / S

                    M_nm = (8 / 3) * rho0 * a**3 * moyenne**2

                else:
                    gamma_nm = 0.3 / m
                    M_nm = gamma_nm * (8 / 3) * rho0 * a**3

                M_tot += M_nm
                Omega_nm = np.sqrt(K_tot / M_tot)

                if m == 0:
                    R_nm = (rho0 * np.pi * a**4 / (2 * c0)) * Omega_nm**2 * moyenne**2

                R_tot = eta * K_tot + 2 * beta * rhoS * N_nm + R_nm

                delta_nm = R_tot / (2 * M_tot)

                omega_d = np.sqrt(Omega_nm**2 - delta_nm**2)

            if m == 0 or n == 1:
                print(f"Mode ({n}, {m}): {phi(k_nm, 0, 0, 0, 0, a, cuve)}")
                print(f"Mode ({n}, {m}): k_nm={k_nm:.4f}, omega_nm={omega_nm:.4f}, delta_n={delta_nm:.4f}, omega_d={omega_d:.4f} | M_tot = {M_tot:.4f} | R_tot = {R_tot:.4f} | N_nm = {N_nm:.4f}")

            # Amplitudes d'excitation et d'écoute
            amp_exc = phi(k_nm, m, r_0, theta_0, theta_0, a, cuve)

            # Amplitude temporelle
            A_nm = (F_0 * amp_exc) / (M_tot * omega_d)
            # if m == 0 or n == 1:
            #     print(f"Mode ({n}, {m}): A_nm={A_nm:.4f}")
            q_nm = A_nm * np.exp(-delta_nm * t) * np.sin(omega_d * t)

            # Champ acoustique rayonné en (r_lst, theta_lst, phi_lst)
            if lst_membrane or not rayonnement:
                amp_ecoute = phi(k_nm, m, r_lst, theta_lst, theta_0, a, cuve)
                signal_total += amp_ecoute * q_nm
            else:
                # Champ acoustique rayonné en (r_lst, theta_lst, phi_lst)
                if lst_membrane or not rayonnement:
                    amp_ecoute = phi(k_nm, m, r_lst, theta_lst, theta_0, a, cuve)
                    signal_total += amp_ecoute * q_nm
                else:
                    q_vel = - q_nm * omega_d**2
                    phi_grille = phi(k_nm, m, grille_r, grille_theta, theta_0, a, cuve)
                    
                    # Critère de transition de Fresnel/Fraunhofer: x = L^2 / 4*lambda
                    longueur_onde = (2 * np.pi * c0) / omega_d
                    dimension_L = 2 * a
                    distance_transition = (dimension_L**2) / (4 * longueur_onde)
                    
                    # Sélection du modèle selon la distance d'observation r_lst
                    if r_lst > distance_transition:
                        distances_d_i = distances_fraunhofer
                        mode_radiatif = "Fraunhofer"
                    else:
                        distances_d_i = distances_fresnel
                        mode_radiatif = "Fresnel"
                        
                    retards_echantillons = np.round((distances_d_i / c0) * fs).astype(int)
                    retards_uniques = np.unique(retards_echantillons)
                    
                    # Amplitude de la pression avec décroissance en 1/r
                    poids_int = (rho0 / (2 * np.pi)) * phi_grille * (dSi / distances_d_i)

                    for retard in retards_uniques:
                        poids_cumule = np.sum(poids_int[retards_echantillons == retard])
                        if np.abs(poids_cumule) < 1e-12:
                            continue
                        if retard > 0:
                            q_retarde = np.pad(q_vel, (retard, 0))[:-retard]
                        else:
                            q_retarde = q_vel

                        signal_total += poids_cumule * q_retarde

            # print(f"Mode (n={n}, m={m}): k_nm={k_nm:.4f}, omega_nm={omega_nm:.4f}, delta_n={delta_n:.4f}, omega_d={omega_d:.4f}, A_nm={A_nm:.4f}, q_nm={q_nm[0]:.4f}")

    # Normalisation
    # pic_max = np.max(np.abs(signal_total))
    # if pic_max > 0:
    #     signal_audio = signal_total / pic_max
    # else:
    #     signal_audio = signal_total

    gain_fixe = 0.001
    signal_audio = signal_total * gain_fixe
    signal_audio = np.clip(signal_audio, -1.0, 1.0)

    return t, signal_audio

if __name__ == "__main__":

    config_path = Path(__file__).with_name("config.json")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    geometry = config["geometry"]
    material = config["material"]
    synthesis = config["synthesis"]
    playing = config["playing"]
    model = config["model"]

    amortissement = model["damping"] # 1 si amortissement, 0 sinon
    cuve = model["coupling"]
    rayonnement = model["radiation"]

    # Coefficients de Rayleigh pour l'amortissement
    eta = 0.0
    beta = 0.0

    if amortissement:
        beta = material["beta"]
        eta = material["eta"]

    a = geometry["radius"]
    V0 = geometry["volume"]
    T = material["tension"]
    rhoS = material["surface_mass"]
    rho0 = material["rho_air"]
    c0 = material["c_air"]
    c = np.sqrt(T / rhoS) # Vitesse de propagation des ondes transverses sur la membrane

    duree = synthesis["duration"]
    fs = synthesis["sampling_frequency"]
    N_modes = synthesis["modes"]
    
    # fmax = fs / 2.0
    # Nmax, Mmax = get_nm(fmax, a, c)

    # if N_modes > Nmax:
    #     print(f"Attention: le nombre de modes demandé ({N_modes}) dépasse le nombre de modes audibles ({Nmax}).")
    #     print(f"Le nombre de modes sera limité à {Nmax}.")
    #     N_modes = Nmax

    M_modes = N_modes
    
    # print(f"a = {a}, c = {c}, fmax = {fmax} -> N_modes = {Nmax}, M_modes = {Mmax}")

    r_0, theta_0 = playing["r_0"] * a, playing["theta_0"] * a
    F_0 = playing["exc_force"]
    r_lst, theta_lst, phi_lst = playing["r_lst"], playing["theta_lst"], playing["phi_lst"]
    lst_membrane = playing["lst_membrane"]  # True si on écoute la membrane, False si on écoute le champ acoustique rayonné

    # print parameters
    print(f"Paramètres de synthèse :")
    print(f"  Rayon de la membrane : {a} m")
    print(f"  Vitesse de propagation : {c:.2f} m/s")
    print(f"  Amortissement : {'activé' if amortissement else 'désactivé'} (beta={beta}, eta={eta})")
    print(f"  Couplage à la cuve : {'activé' if cuve else 'désactivé'}")
    print(f"  Rayonnement acoustique : {'activé' if rayonnement else 'désactivé'}")
    print(f"  Durée de la synthèse : {duree} s")
    print(f"  Fréquence d'échantillonnage : {fs} Hz")
    print(f"  Nombre de modes : {N_modes} (M_modes={M_modes})")
    print(f"  Position d'excitation : r_0={r_0:.3f} m, theta_0={theta_0:.3f} rad")
    print(f"  Force d'excitation : F_0={F_0:.3f} N")
    print(f"  Position de fin d'excitation : r_lst={r_lst:.3f} m, theta_lst={theta_lst:.3f} rad, phi_lst={phi_lst:.3f} rad")

    # # Lancement du calcul
    temps, audio = synthese_membrane(a, c, beta, eta, V0, rhoS, rho0, c0, r_0, theta_0, F_0, r_lst, theta_lst, phi_lst, duree, fs, N_modes, M_modes, cuve, rayonnement, lst_membrane)
    
    # Export audio
    output_directory = Path("output")
    output_directory.mkdir(exist_ok=True)
    cuve_str = "coupling" if cuve else ""
    damp_str = "damping" if amortissement else ""
    ray_str = "radiation" if rayonnement else ""
    write(output_directory / f"timbale_{N_modes}_{damp_str}_{cuve_str}_{ray_str}.wav", fs, audio.astype(np.float32))

    k_n0 = calculer_kn_m0(a, c**2 * rhoS, V0, rho0, c0, N_modes)

    plot_modes(
        a,
        save_path=output_directory / "modes_visualisations.png",
        cuve=cuve,
        k_n0=k_n0 if cuve else None
    )