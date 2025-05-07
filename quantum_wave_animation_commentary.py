
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

st.set_page_config(page_title="Quantum Walk: Animation + Commentary", layout="wide")
st.title("⚛️ Quantum Wavefunction: Animation, Commentary & 3D Visualization")

st.markdown("""
This upgraded app lets you:
- 🎬 **Play/Pause** the animation of quantum evolution.
- 💬 **Get commentary** when paused, explaining the current distribution.
- 🌈 See a **side-by-side 3D heatmap** showing the full time evolution.

---
""")

# Constants
N = 5
MAX_STEPS = 1000

# Quantum evolution matrix
theta = np.pi / 4
U = np.zeros((N, N), dtype=complex)
for i in range(N):
    U[i, i] = np.cos(theta)
    if i > 0:
        U[i, i - 1] = 1j * np.sin(theta)
    if i < N - 1:
        U[i, i + 1] = 1j * np.sin(theta)

# Sidebar controls
start_pos = st.sidebar.slider("Starting Position", 0, N - 1, 2)
total_steps = st.sidebar.slider("Max Steps to Simulate", 1, MAX_STEPS, 200)

# Store all steps of |psi|^2
psi = np.zeros((N, total_steps + 1), dtype=complex)
psi[start_pos, 0] = 1.0

for t in range(1, total_steps + 1):
    psi[:, t] = U @ psi[:, t - 1]

prob_density = np.abs(psi) ** 2  # Shape (N, total_steps+1)

# Animation state
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'running' not in st.session_state:
    st.session_state.running = False

# --- Layout: side by side ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎬 Animated Quantum Probability Distribution")

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.bar(np.arange(N), prob_density[:, st.session_state.frame])
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Probability")
    ax1.set_title(f"|ψ(x)|² at t = {st.session_state.frame}")
    st.pyplot(fig1)

    colA, colB = st.columns(2)
    if colA.button("⏯ Play/Pause"):
        st.session_state.running = not st.session_state.running
    if colB.button("⏮ Restart"):
        st.session_state.frame = 0
        st.session_state.running = False

    # Auto-advance if running
    if st.session_state.running and st.session_state.frame < total_steps:
        time.sleep(0.1)
        st.session_state.frame += 1
        st.experimental_rerun()

    # Commentary
    if not st.session_state.running:
        current_probs = prob_density[:, st.session_state.frame]
        uniformity = np.allclose(current_probs, np.ones(N) / N, atol=0.05)
        peak_idx = np.argmax(current_probs)
        max_val = current_probs[peak_idx]

        st.subheader("💬 Commentary")
        if uniformity:
            st.write("🔍 **The distribution is nearly uniform.** This occurs because destructive interference balances out the amplitudes, flattening the probability density temporarily.")
        elif max_val > 0.7:
            st.write(f"🔍 **The wavefunction is strongly localized at position {peak_idx}.** This is a sign of constructive interference concentrating probability.")
        else:
            st.write("🔍 **The wavefunction shows a dynamic, oscillatory pattern.** Probability is spread out but not yet uniform, reflecting ongoing quantum interference.")

with col2:
    st.subheader("🌈 3D Heatmap: |ψ(x, t)|²")

    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(111, projection='3d')

    X = np.arange(N)
    T = np.arange(total_steps + 1)
    X_grid, T_grid = np.meshgrid(X, T)

    Z = prob_density.T  # shape (steps+1, N)

    surf = ax2.plot_surface(T_grid, X_grid, Z, cmap=cm.viridis, linewidth=0, antialiased=False)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position")
    ax2.set_zlabel("Probability")
    ax2.set_title("Probability Density |ψ(x, t)|² Over Time")
    ax2.view_init(elev=30, azim=225)
    fig2.colorbar(surf, shrink=0.5, aspect=5)
    st.pyplot(fig2)
