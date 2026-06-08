# Physics-Informed Deep Learning for Multi-Target Ballistic Trajectory Forecasting

### Abstract
This article details the architecture of a real-time predictive tracking system designed for dense, multi-target ballistic environments. We integrate Sequence-to-Sequence (Seq2Seq) neural networks with Hungarian-based kinematic assignment to ensure sub-pixel accuracy and stable target identification.

---

### 1. Introduction & State Estimation
Tracking high-velocity projectiles in real-time presents a severe geometric challenge. Traditional distance-based tracking algorithms fail during trajectory intersections, leading to critical identity switches.

* **Kinematic Prior:** Modeling inertia using Kalman-derived principles.
* **Global Optimization:** Resolving crossings via the Hungarian (Kuhn-Munkres) algorithm, ensuring zero identity switches in dense clusters.

### 2. Neural Architecture: The Seq2Seq GRU
The forecasting module leverages an autoregressive GRU (Gated Recurrent Unit) to model long-term non-linear kinematic dependencies.

| Module | Logic |
| :--- | :--- |
| **Input** | 30-frame observation window $(x, y)$ |
| **Embedding** | Relative centering (origin shift to $p_{t=30}$) |
| **Encoder** | Projection to 256-dim latent space via GRU |
| **Decoder** | Autoregressive delta prediction ($\Delta\hat{p}$) |

### 3. Physics-Informed Loss (Sniper Horizon Loss)
To mathematically enforce Newtonian realism, we define the objective as a composite sum:

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{terminal}}+\lambda_{\text{acc}}\mathcal{L}_{\text{acc}}$$

* **$\mathcal{L}_{\text{traj}}$**: Temporally-weighted MAE for structural curve integrity.
* **$\mathcal{L}_{\text{terminal}}$**: High-weight penalty for precise impact coordinates.
* **$\mathcal{L}_{\text{acc}}$**: Acceleration variance penalty (enforcing constant-gravity motion).

---

### 4. Empirical Results
The system transcends simple pattern matching by retro-engineering the physical laws of ballistics.

* **[Demo 1: Tracking & Assignment Performance](https://github.com/user-attachments/assets/e51d8afc-254c-4141-bda0-33e45940fbdb)**
* **[Demo 2: Full HUD System Forecasting](https://github.com/user-attachments/assets/f32a3b89-02a1-4912-a816-164cd364745d)**

### 5. Conclusion
By embedding kinematic constraints directly into the loss landscape, the system is robust to occlusions, fully translation-invariant, and highly scalable for real-time defense applications.

---
*© 2026 Raphael KHAYAT. All Rights Reserved. Source code available upon request.*
