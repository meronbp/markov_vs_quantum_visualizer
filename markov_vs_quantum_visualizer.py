
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Markov vs Quantum Evolution", layout="wide")
st.title("🔄 Markov vs ⚛️ Quantum Evolution")

st.markdown("""
This app compares:
- **Markov Chain Evolution:** Classical probability distribution \( P(t) \)
- **Quantum Evolution:** Wavefunction \( \psi(t) \) amplitudes

We use a **simple 5-point lattice** for both cases, but:
- The **Markov** case uses a **stochastic matrix** (probabilities spread & settle).
- The **Quantum** case uses a **unitary matrix** (amplitudes oscillate forever).

---
""")

# Markov transition matrix (5-point reflective random walk)
U_markov = np.array([
    [0.5, 0.5, 0,   0,   0],
    [0.5, 0,   0.5, 0,   0],
    [0,   0.5, 0,   0.5, 0],
    [0,   0,   0.5, 0,   0.5],
    [0,   0,   0,   0.5, 0.5]
])

# Quantum evolution: use a discrete-time "tight-binding" unitary evolution
# A simple symmetric unitary that mimics hopping between neighbors
theta = np.pi / 4  # hopping angle
U_quantum = np.zeros((5, 5), dtype=complex)
for i in range(5):
    U_quantum[i, i] = np.cos(theta)
    if i > 0:
        U_quantum[i, i-1] = 1j * np.sin(theta)
    if i < 4:
        U_quantum[i, i+1] = 1j * np.sin(theta)

# Simulation controls
steps = st.slider("Number of time steps", 1, 100, 30)
start_pos = st.slider("Starting position (0 to 4)", 0, 4, 2)

# Initial states
P0 = np.zeros(5)
P0[start_pos] = 1

psi0 = np.zeros(5, dtype=complex)
psi0[start_pos] = 1.0 + 0j

# Evolve
P_t = P0.copy()
psi_t = psi0.copy()

for t in range(steps):
    P_t = U_markov @ P_t
    psi_t = U_quantum @ psi_t

# Set up side-by-side plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Markov Chain Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(range(5), P_t.real)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"P(t={steps})")
    ax1.set_xticks(range(5))
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)

with col2:
    st.subheader("⚛️ Quantum Wavefunction Amplitudes")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(5), np.abs(psi_t)**2)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("|ψ|² (Probability Density)")
    ax2.set_title(f"|ψ(t={steps})|²")
    ax2.set_xticks(range(5))
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

st.markdown("---")
st.caption("Note: Markov evolves toward a flat distribution. Quantum evolution keeps oscillating forever (unless measured).")
