# Ballistic Trajectory Forecasting System

### Abstract
This project presents a real-time predictive tracking system for dense, multi-target environments. It integrates Sequence-to-Sequence (Seq2Seq) neural networks with Hungarian-based kinematic assignment to ensure sub-pixel accuracy and stable target identification.

---

### 1. The Tracking Framework
The system uses the Hungarian (Kuhn-Munkres) algorithm for global assignment. This eliminates track fragmentation during projectile crossings.
* **Kinematic Prior:** Kalman-based inertia modeling.
* **Global Optimization:** Minimize assignment cost to prevent identity switches.

### 2. Deep Learning Architecture
The forecasting module uses an autoregressive GRU (Gated Recurrent Unit) to map relative coordinates to future trajectories.
* **Relative Centering:** Anchoring sequences at $p_t$ to ensure spatial invariance.
* **Autoregressive Decoding:** Step-by-step prediction of future positional deltas.

### 3. Physics-Informed Loss (Sniper Horizon Loss)
We enforce physical realism by calculating the total loss as:
$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{terminal}}+\lambda_{\text{acc}}\mathcal{L}_{\text{acc}}$$
This ensures the model respects constant-gravity parabolic motion.

---

### 4. Demonstrations (Watch on GitHub)
If the player below does not load, use these direct links:

* **[Demo 1: Multi-Target Tracking Performance](https://github.com/user-attachments/assets/e51d8afc-254c-4141-bda0-33e45940fbdb)**
* **[Demo 2: Full HUD Forecasting System](https://github.com/user-attachments/assets/f32a3b89-02a1-4912-a816-164cd364745d)**

---

### 5. Contact & Copyright
© 2026 [Raphael KHAYAT]. All Rights Reserved.
*For professional inquiries, source code and training architecture are available upon request during technical interviews.*
