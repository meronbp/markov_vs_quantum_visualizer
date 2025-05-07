
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

st.set_page_config(page_title="Advanced Markov vs Quantum Visualizer", layout="wide")
st.title("🚀 Markov vs Quantum Evolution: Advanced Visualizer")

st.markdown("""
This app shows:
- **Left:** Markov chain probability \( P(t) \)
- **Right:** Quantum wavefunction probability density \( |\psi(t)|^2 \)

✅ **Features:**
- 🎬 **Play/Pause Animation** to see the evolution dynamically.
- 🌋 **3D Heatmap**: visualize how probabilities evolve over time.
- 🚀 **Up to 1000 steps** for deep exploration.

---
""")

# Set up matrices
size = 5  # lattice points

U_markov = np.array([
    [0.5, 0.5, 0,   0,   0],
    [0.5, 0,   0.5, 0,   0],
    [0,   0.5, 0,   0.5, 0],
    [0,   0,   0.5, 0,   0.5],
    [0,   0,   0,   0.5, 0.5]
])

theta = np.pi / 4
U_quantum = np.zeros((size, size), dtype=complex)
for i in range(size):
    U_quantum[i, i] = np.cos(theta)
    if i > 0:
        U_quantum[i, i-1] = 1j * np.sin(theta)
    if i < size - 1:
        U_quantum[i, i+1] = 1j * np.sin(theta)

# Session state for animation
if "playing" not in st.session_state:
    st.session_state.playing = False
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# Controls
max_steps = st.slider("Maximum number of time steps", 1, 1000, 200)
start_pos = st.slider("Starting position (0 to 4)", 0, 4, 2)

# Initial states
P0 = np.zeros(size)
P0[start_pos] = 1
psi0 = np.zeros(size, dtype=complex)
psi0[start_pos] = 1.0 + 0j

# Play/Pause buttons
col_play, col_reset = st.columns(2)
if col_play.button("▶️ Play" if not st.session_state.playing else "⏸ Pause"):
    st.session_state.playing = not st.session_state.playing
if col_reset.button("⏮ Reset"):
    st.session_state.current_step = 0
    st.session_state.playing = False

# Run simulation up to max_steps
P_history = [P0.copy()]
psi_history = [psi0.copy()]
for t in range(1, max_steps + 1):
    P_next = U_markov @ P_history[-1]
    psi_next = U_quantum @ psi_history[-1]
    P_history.append(P_next)
    psi_history.append(psi_next)

# Advance animation frame
if st.session_state.playing and st.session_state.current_step < max_steps:
    st.session_state.current_step += 1
    time.sleep(0.05)

current_step = st.session_state.current_step

# Plot side-by-side graphs
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"🔄 Markov Chain P(t={current_step})")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(range(size), P_history[current_step])
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Probability")
    ax1.set_title("Markov Chain Distribution")
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)

with col2:
    st.subheader(f"⚛️ Quantum |ψ(t={current_step})|²")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(range(size), np.abs(psi_history[current_step])**2)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("|ψ|² (Probability Density)")
    ax2.set_title("Quantum Probability Density")
    ax2.set_ylim(0, 1)
    st.pyplot(fig2)

# 3D heatmap
st.subheader("🌋 3D Heatmap of Quantum Probability Evolution")
fig3 = plt.figure(figsize=(10, 6))
ax3 = fig3.add_subplot(111, projection='3d')
X, Y = np.meshgrid(range(size), range(current_step + 1))
Z = np.array([np.abs(psi_history[t])**2 for t in range(current_step + 1)])
ax3.plot_surface(X, Y, Z, cmap='viridis')
ax3.set_xlabel("Position")
ax3.set_ylabel("Time Step")
ax3.set_zlabel("Probability Density")
ax3.set_title("Quantum Evolution Heatmap")
st.pyplot(fig3)

st.markdown("---")
st.caption("Use Play/Pause to animate. The 3D plot updates in real time! 🚀")
