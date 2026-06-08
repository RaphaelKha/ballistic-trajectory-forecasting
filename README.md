# Physics-Informed Deep Learning for Multi-Target Ballistic Trajectory Forecasting

### Abstract
This article details the architecture of a real-time predictive tracking system designed for dense, multi-target ballistic environments. We combine robust synthetic data generation, spatial invariance, and a physics-informed loss function to achieve sub-pixel forecasting accuracy and eliminate identity switches during target occlusion.

---

### 1. The Tracking Engine: Hungarian Assignment
To provide our neural network with pure, unbroken sequential data, we implemented a Simple Online and Realtime Tracking (SORT) framework. We resolve crossing trajectories using the Hungarian algorithm, ensuring 0 identity switches.

![Tracking Demo](https://github.com/user-attachments/assets/e51d8afc-254c-4141-bda0-33e45940fbdb)

---

### 2. Neural Architecture: The Seq2Seq GRU
We utilize a sequence-to-sequence GRU architecture to map relative projectile coordinates to future trajectories.

![Architecture GRU](architecture.png)

### 3. Physics-Informed Loss Formulation
We engineered a custom composite objective, the **Sniper Horizon Loss**, to mathematically enforce physical realism:

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{terminal}}+\lambda_{\text{acc}}\mathcal{L}_{\text{acc}}$$

---

### 4. Performance & HUD
The AI retro-engineers the underlying physical laws, yielding sub-pixel accuracy and projecting reliable "Lead Indicators" for 40+ simultaneous targets.

![Full HUD System](https://github.com/user-attachments/assets/f32a3b89-02a1-4912-a816-164cd364745d)

---

### 5. Conclusion & Availability
The system is robust to occlusions and fully translation-invariant. 
*Copyright & License: © 2026 [Raphael KHAYAT]. All Rights Reserved.*
