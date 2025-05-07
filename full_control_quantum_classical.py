
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

st.set_page_config(page_title="Quantum & Classical Evolution: Full Control", layout="wide")
st.title("⚛️ Quantum vs 🔄 Classical Evolution: Full Animation Control")

st.markdown("""
**Features:**
- 🎬 **Play/Pause button:** Starts/stops automatic simulation from step 0 to 1000.
- 🐢🚀 **Speed control:** Adjust how fast time progresses.
- ✅ **Choose what to display:** Quantum state, Classical state, or both.
- 📊 **Bar charts + 3D heatmaps** side by side, updating in sync.

---
""")

# Constants
N = 5
MAX_STEPS = 1000

# Quantum evolution matrix
theta = np.pi / 4
U_quantum = np.zeros((N, N), dtype=complex)
for i in range(N):
    U_quantum[i, i] = np.cos(theta)
    if i > 0:
        U_quantum[i, i - 1] = 1j * np.sin(theta)
    if i < N - 1:
        U_quantum[i, i + 1] = 1j * np.sin(theta)

# Classical Markov transition matrix
U_classical = np.array([
    [0.5, 0.5, 0,   0,   0],
    [0.5, 0,   0.5, 0,   0],
    [0,   0.5, 0,   0.5, 0],
    [0,   0,   0.5, 0,   0.5],
    [0,   0,   0,   0.5, 0.5]
])

# Sidebar controls
st.sidebar.header("Simulation Settings")
start_pos = st.sidebar.slider("Starting Position", 0, N - 1, 2)
total_steps = st.sidebar.slider("Total Steps to Simulate", 1, MAX_STEPS, 500)
speed = st.sidebar.slider("Speed (Lower is Faster)", 0.01, 1.0, 0.1, 0.01)

st.sidebar.header("Display Options")
show_quantum = st.sidebar.checkbox("Show Quantum Evolution", True)
show_classical = st.sidebar.checkbox("Show Classical Evolution", True)

# Prepare data arrays
psi = np.zeros((N, total_steps + 1), dtype=complex)
P_classical = np.zeros((N, total_steps + 1))

# Initial conditions
psi[start_pos, 0] = 1.0
P_classical[start_pos, 0] = 1.0

for t in range(1, total_steps + 1):
    psi[:, t] = U_quantum @ psi[:, t - 1]
    P_classical[:, t] = U_classical @ P_classical[:, t - 1]

prob_quantum = np.abs(psi) ** 2  # (N, total_steps+1)

# --- Session state management ---
if 'frame' not in st.session_state:
    st.session_state.frame = 0
if 'running' not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([1, 1])

# LEFT: Bar Charts
with col1:
    st.subheader("📊 Probability Distributions (Bar Charts)")
    fig, ax = plt.subplots(figsize=(6, 4))

    if show_quantum:
        ax.bar(np.arange(N) - 0.2, prob_quantum[:, st.session_state.frame],
               width=0.4, label="Quantum")
    if show_classical:
        ax.bar(np.arange(N) + 0.2, P_classical[:, st.session_state.frame],
               width=0.4, label="Classical")

    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_title(f"t = {st.session_state.frame}")
    ax.set_xticks(range(N))
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

# RIGHT: 3D Heatmaps
with col2:
    st.subheader("🌈 3D Heatmaps: |ψ(x, t)|² and Classical P(x, t)")

    fig2 = plt.figure(figsize=(8, 5))
    ax2 = fig2.add_subplot(111, projection='3d')

    X = np.arange(N)
    T = np.arange(total_steps + 1)
    X_grid, T_grid = np.meshgrid(X, T)

    if show_quantum:
        Z_q = prob_quantum.T
        ax2.plot_surface(T_grid, X_grid, Z_q, cmap=cm.viridis, alpha=0.7, linewidth=0, antialiased=False, label="Quantum")

    if show_classical:
        Z_c = P_classical.T
        ax2.plot_surface(T_grid, X_grid, Z_c, cmap=cm.plasma, alpha=0.7, linewidth=0, antialiased=False, label="Classical")

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position")
    ax2.set_zlabel("Probability")
    ax2.set_title("Probability Evolution Over Time")
    ax2.view_init(elev=30, azim=225)
    st.pyplot(fig2)

# --- Play/Pause Control ---
st.subheader("🎬 Control Panel")
colA, colB, colC = st.columns(3)
if colA.button("▶️ Play" if not st.session_state.running else "⏸ Pause"):
    st.session_state.running = not st.session_state.running
if colB.button("⏮ Restart"):
    st.session_state.frame = 0
    st.session_state.running = False
colC.write(f"Current Time Step: {st.session_state.frame}")

# Auto-advance if running
if st.session_state.running and st.session_state.frame < total_steps:
    time.sleep(speed)
    st.session_state.frame += 1
    st.experimental_rerun()
