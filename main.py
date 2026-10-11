import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# On importe les fonctions du fichier de synthèse et des utilitaires
import synthese
from utils import plot_modes, calculer_kn_m0

st.markdown("""
<style>
    /* Réduit l'écart entre chaque élément vertical de la sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important; 
    }
    /* Réduit le padding intérieur en haut de la sidebar */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Synthèse Membrane (Timbale) - GUI")

with st.sidebar:
    st.header("Paramètres")
    
    st.subheader("Modèle")
    amortissement = st.checkbox("Amortissement", value=True, key="amortissement")
    cuve = st.checkbox("Couplage cuve", value=True, key="cuve")
    rayonnement = st.checkbox("Rayonnement acoustique", value=True, key="rayonnement")

    st.subheader("Géométrie")
    a = st.number_input("Rayon a (m)", value=0.2, format="%.3f", step=0.01, key="a")
    V0 = st.number_input("Volume cuve V0 (m³)", value=0.05, format="%.3f", step=0.01, key="V0")

    st.subheader("Matériau")
    T = st.number_input("Tension T (N/m)", value=3500.0, step=100.0, key="T")
    rhoS = st.number_input("Masse surfacique ρ_S (kg/m²)", value=0.4, step=0.01, key="rhoS")
    rho0 = st.number_input("Densité air ρ_0 (kg/m³)", value=1.204, step=0.01, key="rho0")
    c0 = st.number_input("Célérité air c_0 (m/s)", value=343.0, step=1.0, key="c0")
    beta = st.number_input("Frottement visqueux β", value=5.0, step=0.1, disabled=not amortissement, key="beta")
    eta = st.number_input("Viscoélasticité η", value=1e-6, format="%e", step=1e-7, disabled=not amortissement, key="eta")

    st.subheader("Excitation")
    F_0 = st.number_input("Force d'excitation F_0 (N)", value=1.0, step=0.1, key="F_0")

    st.subheader("Écoute")
    lst_membrane = st.checkbox("Écoute directe sur la membrane", value=False, key="lst_membrane")
    if not lst_membrane:
        st.write("*Paramètres pour le champ rayonné (3D):*")
        r_lst = st.number_input("Distance d'écoute r (m)", value=0.5, step=0.1, key="r_lst")
        theta_lst = st.slider("Azimut θ (rad)", min_value=-np.pi, max_value=np.pi, value=0.0, key="theta_lst")
        phi_lst = st.slider("Élévation φ (rad)", min_value=0.0, max_value=np.pi, value=0.785, key="phi_lst")
    else:
        st.write("*Paramètres définis via le clic sur l'image*")
        r_lst, theta_lst, phi_lst = 0.0, 0.0, 0.0  # Seront remplacés par le clic

    st.subheader("Synthèse")
    duree = st.number_input("Durée de la synthèse (s)", value=2.0, step=0.1, key="duree")
    fs = st.number_input("Fréquence d'échantillonnage (Hz)", value=44100, step=1000, key="fs")
    N_modes = st.number_input("Nombre de modes (N = M)", value=16, min_value=1, max_value=128, step=1, key="N_modes")


# Vitesses de propagation
c = np.sqrt(T / rhoS)

# Initialisation des positions relatives sur la membrane (r_ratio ∈ [0, 1], theta ∈ [-pi, pi])
if "r_0_ratio" not in st.session_state:
    st.session_state.r_0_ratio = 0.1 / 0.2  # Valeurs du JSON : r_0=0.1 pour a=0.2 -> ratio = 0.5
if "theta_0" not in st.session_state:
    st.session_state.theta_0 = 0.0
if "r_lst_ratio" not in st.session_state:
    st.session_state.r_lst_ratio = 0.5
if "theta_lst_2D" not in st.session_state:
    st.session_state.theta_lst_2D = 0.0
if "last_click" not in st.session_state:
    st.session_state.last_click = None

st.header("Positions sur la membrane")

# Sélection du type de clic si l'écoute sur la membrane est activée
options_clic = ["Excitation"]
if lst_membrane:
    options_clic.append("Écoute")
    
click_type = st.radio(
    "Sélectionner le paramètre à placer :",
    options=options_clic,
    horizontal=True
)

# ---- DESSIN DE LA MEMBRANE ----
IMG_SIZE = 400
img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "#787171")
draw = ImageDraw.Draw(img)

center = IMG_SIZE // 2
radius_px = (IMG_SIZE // 2) - 20

# Cercle principal (membrane)
draw.ellipse(
    [(center - radius_px, center - radius_px), (center + radius_px, center + radius_px)],
    fill="#e0e0e0", outline="#000000", width=3
)
# Croix centrale pour repère
draw.line([(center-10, center), (center+10, center)], fill="#aaaaaa", width=1)
draw.line([(center, center-10), (center, center+10)], fill="#aaaaaa", width=1)

# Placer le point d'excitation
if st.session_state.r_0_ratio is not None:
    x_px = center + int(st.session_state.r_0_ratio * radius_px * np.cos(st.session_state.theta_0))
    y_px = center + int(st.session_state.r_0_ratio * radius_px * np.sin(st.session_state.theta_0))
    draw.ellipse([(x_px-6, y_px-6), (x_px+6, y_px+6)], fill="blue", outline="black")

# Placer le point d'écoute (uniquement si sur la membrane)
if lst_membrane and st.session_state.r_lst_ratio is not None:
    x_px = center + int(st.session_state.r_lst_ratio * radius_px * np.cos(st.session_state.theta_lst_2D))
    y_px = center + int(st.session_state.r_lst_ratio * radius_px * np.sin(st.session_state.theta_lst_2D))
    draw.ellipse([(x_px-6, y_px-6), (x_px+6, y_px+6)], fill="red", outline="black")

st.write("Cliquez sur le cercle pour sélectionner la position de l'impact (bleu) ou de l'écoute (rouge).")
clic = streamlit_image_coordinates(img, key="membrane_click")

# Gestion du clic
if clic is not None and clic != st.session_state.last_click:
    st.session_state.last_click = clic
    dx = clic['x'] - center
    dy = clic['y'] - center
    r_px = np.sqrt(dx**2 + dy**2)
    
    # On limite le clic au rayon de la membrane
    r_ratio = min(r_px / radius_px, 1.0)
    theta = np.arctan2(dy, dx)
    
    if click_type == "Excitation":
        st.session_state.r_0_ratio = r_ratio
        st.session_state.theta_0 = theta
    elif click_type == "Écoute":
        st.session_state.r_lst_ratio = r_ratio
        st.session_state.theta_lst_2D = theta

    st.rerun()

st.divider()
col1, col2 = st.columns(2)

with col1:
    r_0_phys = st.session_state.r_0_ratio * a
    st.info(f"**Excitation :** r = {r_0_phys:.3f} m, θ = {st.session_state.theta_0:.2f} rad")
        
with col2:
    if lst_membrane:
        r_lst_phys = st.session_state.r_lst_ratio * a
        theta_lst_phys = st.session_state.theta_lst_2D
        st.error(f"**Écoute (Membrane) :** r = {r_lst_phys:.3f} m, θ = {theta_lst_phys:.2f} rad")
    else:
        r_lst_phys = r_lst
        theta_lst_phys = theta_lst
        st.error(f"**Écoute (Champ 3D) :** r = {r_lst_phys:.2f}m, θ = {theta_lst:.2f}, φ = {phi_lst:.2f}")

st.header("Résultats")

if st.button("SIMULATION", type="primary"):
    
    with st.spinner("Synthèse en cours..."):
        
        beta_val = beta if amortissement else 0.0
        eta_val = eta if amortissement else 0.0
        M_modes = N_modes

        # Lancement de la synthèse
        temps, audio = synthese.synthese_membrane(
            a, c, beta_val, eta_val, V0, rhoS, rho0, c0, 
            r_0_phys, st.session_state.theta_0, F_0, 
            r_lst_phys, theta_lst_phys, phi_lst, 
            duree, fs, N_modes, M_modes, 
            cuve, rayonnement, lst_membrane
        )
        
        # Normalisation du signal audio pour le lecteur Streamlit
        max_amp = np.max(np.abs(audio))
        audio_norm = audio / max_amp if max_amp > 0 else audio
        audio_norm = np.float32(audio_norm)

    st.success("Terminé.")

    # Lecteur audio
    st.subheader("Audio")
    st.audio(audio_norm, sample_rate=fs)

    # Forme d'onde & spectrogramme
    st.subheader("DSP")
    
    col_onde, col_spec = st.columns(2)
    
    with col_onde:
        st.markdown("**Forme d'onde**")
        fig_wave, ax_wave = plt.subplots(figsize=(6, 4))
        ax_wave.plot(temps, audio, color='#1f77b4', linewidth=0.8)
        ax_wave.set_xlabel("Temps (s)")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.grid(True, alpha=0.3)
        st.pyplot(fig_wave)
        
    with col_spec:
        st.markdown("**Spectrogramme**")
        fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
        Pxx, freqs, bins, im = ax_spec.specgram(audio, Fs=fs, NFFT=2048, cmap='viridis')
        ax_spec.set_xlabel("Temps (s)")
        ax_spec.set_ylabel("Fréquence (Hz)")
        fig_spec.colorbar(im, ax=ax_spec, label="Intensité (dB)")
        st.pyplot(fig_spec)

    # Déformées
    st.subheader("Déformées modales")
    with st.spinner("Génération des déformées..."):
        output_directory = Path("output")
        output_directory.mkdir(exist_ok=True)
        modes_path = output_directory / "modes_visualisations.png"
        
        # Calcul des racines exactes si la cuve est active
        k_n0 = None
        if cuve:
            k_n0 = calculer_kn_m0(a, c**2 * rhoS, V0, rho0, c0, N_modes)

        plot_modes(a, save_path=modes_path, cuve=cuve, k_n0=k_n0)
        
        if modes_path.exists():
            st.image(str(modes_path), width='stretch')
        else:
            st.error("L'image des modes n'a pas pu être générée.")