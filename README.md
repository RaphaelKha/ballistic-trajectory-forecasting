# Ballistic Trajectory Forecasting System

### Abstract
This article details the architecture of a real-time predictive tracking system designed for dense, multi-target ballistic environments. We integrate Sequence-to-Sequence (Seq2Seq) neural networks with Hungarian-based kinematic assignment to ensure sub-pixel accuracy and stable target identification.

---

### 1. Tracking & The Hungarian Algorithm
The system uses the Hungarian (Kuhn-Munkres) algorithm for global assignment. This eliminates track fragmentation during projectile crossings.
* **Kinematic Prior:** Kalman-based inertia modeling.
* **Global Optimization:** Minimize assignment cost to prevent identity switches.

**Visual Demonstration:**
[![](https://raw.githubusercontent.com/RaphaelKha/ballistic-trajectory-forecasting/main/thumbnail_tracking.png)](https://github.com/user-attachments/assets/e51d8afc-254c-4141-bda0-33e45940fbdb)
*(Click the image to view the tracking performance in full quality)*

---

### 2. Neural Architecture (Seq2Seq GRU)
The forecasting module uses an autoregressive GRU architecture:
1. **Relative Centering:** Anchoring sequences at $p_t$ for spatial invariance.
2. **Latent Encoding:** Projection to 256-dim space via GRU.
3. **Autoregressive Decoding:** Step-by-step delta prediction.

---

### 3. Physics-Informed Loss (Sniper Horizon Loss)
We enforce physical realism by calculating the total loss as:
$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{terminal}}+\lambda_{\text{acc}}\mathcal{L}_{\text{acc}}$$

**Full System HUD:**
[![](https://raw.githubusercontent.com/RaphaelKha/ballistic-trajectory-forecasting/main/thumbnail_hud.png)](https://github.com/user-attachments/assets/f32a3b89-02a1-4912-a816-164cd364745d)
*(Click the image to see the full HUD forecasting system)*

---

### 4. Copyright
© 2026 [Raphael KHAYAT]. All Rights Reserved.
