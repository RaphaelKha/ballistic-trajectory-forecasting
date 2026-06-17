# Physics-Informed Deep Learning for Multi-Target Ballistic Trajectory Forecasting

![Full HUD System](https://github.com/RaphaelKha/ballistic-trajectory-forecasting/blob/main/thumbnail_hud.png?raw=true)
*The system successfully retro-engineers the underlying physical laws, yielding sub-pixel forecasting accuracy and projecting reliable "Lead Indicators" (Red crosses) for 40+ simultaneous projectiles.*

---

### Abstract
This article details the architecture of a real-time predictive tracking system designed for dense, multi-target ballistic environments. While traditional kinematic filters handle initial state estimation, long-term non-linear trajectory forecasting is delegated to a custom Sequence-to-Sequence (Seq2Seq) neural network. By combining robust synthetic data generation, spatial invariance, and a physics-informed loss function, the system achieves sub-pixel forecasting accuracy and eliminates identity switches during target occlusion.

---

### 1. Introduction & State Estimation (The SORT Paradigm)
Tracking dozens of high-velocity projectiles in real-time presents a severe geometric challenge. Traditional "greedy" distance-based tracking algorithms fail during trajectory intersections, leading to critical identity switches (track fragmentation). 

To provide our neural network with pure, unbroken sequential data, we implemented a Simple Online and Realtime Tracking (SORT) framework. 
1. **Kinematic Prior:** We model the physical inertia of each target using principles derived from the Kalman Filter, predicting where a projectile *should* be based on its velocity.
2. **Global Optimization:** We resolve crossing trajectories using the Hungarian (Kuhn-Munkres) algorithm. By minimizing the global assignment cost matrix rather than making localized greedy choices, the system perfectly maintains target identities even in dense clusters.

![Tracking Demonstration](https://github.com/user-attachments/assets/9a0bd1eb-ddc2-409d-ab85-327d84004ce4)
*Notice the `TGT-X` labels: even as multiple projectiles densely cross paths in the center of the screen, the Hungarian assignment matrix ensures 0 identity switches and 0 track fragmentation.*

---

### 2. Overcoming the OOD Problem: Synthetic Kinematic Generation
A predictive model is only as robust as its training distribution. Training exclusively on projectiles originating from a fixed region causes the network to suffer from the Out-of-Distribution (OOD) problem when encountering trajectories in unseen spatial zones. 

To force the network to internalize universal Newtonian mechanics rather than memorizing absolute screen coordinates, we built a randomized synthetic data pipeline governed by discrete kinematic equations.

**2.1 Spatial and Kinematic Generalization**
For each sample, the initial position $p_0$ and velocity $v_0$ are sampled from a uniform distribution spanning the entire canvas space and a wide spectrum of velocities:

$$p_0=(x_0,y_0)\sim\mathcal{U}(0,W)\times\mathcal{U}(0,H)$$

$$v_0=(v_{x0},v_{y0})\sim\mathcal{U}(-v_{max},v_{max})\times\mathcal{U}(-v_{max},v_{min})$$

A full trajectory of length $L$ (e.g., $L=100$ frames) is then generated using discrete state updates with a constant gravity vector $g$:

$$v_{t}=v_{t-1}+(0,g)$$

$$p_{t}=p_{t-1}+v_{t}$$

**2.2 Temporal Generalization (Stochastic Windowing)**
To ensure the model learns to forecast from any point in a projectile's flight (ascent, apogee, or descent), we extract the training window $W$ using a stochastic temporal offset $T_{start}$:

$$T_{start}\sim\mathcal{U}(0,L-(N_{obs}+N_{pred}))$$

The final sequence passed to the model is the subset $W=\{p_{T_{start}},\dots,p_{T_{start}+N_{obs}+N_{pred}}\}$, ensuring the neural network encounters a perfectly uniform distribution of flight phases.

---

### 3. Neural Network Architecture: The Seq2Seq GRU
With pristine data guaranteed, trajectory forecasting is formulated as an autoregressive sequence generation problem. Drawing conceptual inspiration from modern State Space Models (SSM) and architectures like **VideoMamba**, our sequence-to-sequence formulation effectively captures long-term kinematic dependencies while maintaining the ultra-low latency required for real-time inference.

**3.1 Translation Invariance (Relative Centering)**
Before entering the network, the $30$-frame observation sequence undergoes strict relative centering. The sequence is anchored to its most recent observation, shifting the coordinate origin to $(0,0)$. This mathematical transformation is the key to true spatial invariance.

**3.2 The Encoder (Latent Projection)**
The normalized relative coordinates are passed through a fully connected layer to project the 2D spatial data into a high-dimensional latent space. The GRU encoder then processes this sequence:

$$h_t=\text{GRU}(\text{ReLU}(W_{\text{in}}x_t+b_{\text{in}}),h_{t-1})$$

The final hidden state $h_T$ acts as a compressed "kinematic memory," encoding the projectile's complete velocity, acceleration, and curvature profile.

**3.3 The Autoregressive Decoder & Data Flow**
For the forecasting phase, the model operates autoregressively over the prediction horizon $N=30$. At each future timestep, the network utilizes the evolving hidden state to predict a positional delta $\Delta\hat{p}$.

* 📥 **Input:** `[Absolute Sequence 30x2]`
* 🎯 **Preprocessing:** `[Relative Centering: p_t - p_30]`
* 🧠 **Embedding:** `[Linear Projection Layer: 2 → 256]`
* ⚙️ **Encoding:** `[GRU Encoder Cell]` ➔ *(Yields Final Kinematic State h_30)*
* 🔄 **Decoding:** `[Autoregressive Decoder]` ➔ *(Loops 30x to predict Deltas)*
* 🌍 **Postprocessing:** `[Absolute Re-anchoring]`
* 📤 **Output:** `[Future Trajectory 30x2]`

---

### 4. Physics-Informed Custom Loss Formulation
Standard regression metrics like Mean Squared Error (MSE) treat all positional errors equally. However, ballistic forecasting requires specific kinematic constraints. We engineered a custom composite objective, the Sniper Horizon Loss, to mathematically enforce physical realism.

**4.1 Temporally-Weighted Trajectory Loss ($\mathcal{L}_{\text{traj}}$)**
We apply a linearly increasing temporal weight vector $w_t$ to the MAE of the trajectory, prioritizing the structural integrity of the long-term curve.

$$\mathcal{L}_{\text{traj}}=\frac{1}{N}\sum_{t=1}^{N}w_t\|\hat{p}_t-p_t\|_1$$

**4.2 Terminal Impact Loss ($\mathcal{L}_{\text{terminal}}$)**
We apply an extreme penalty multiplier $\lambda_{\text{term}}$ to the final frame prediction error to guarantee precise impact coordinate acquisition.

$$\mathcal{L}_{\text{terminal}}=\lambda_{\text{term}}\|\hat{p}_N-p_N\|_1$$

**4.3 Kinematic Acceleration Penalty ($\mathcal{L}_{\text{acc}}$)**
We penalize the variance of the predicted acceleration $a_t$, ensuring a smooth flight path dictated by constant gravity.

$$\mathcal{L}_{\text{acc}}=\mathbb{E}[\|a_t\|_2^2]$$

**Final Objective:**

$$\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{terminal}}+\lambda_{\text{acc}}\mathcal{L}_{\text{acc}}$$

**4.4 Quantitative Convergence**
The model rapidly breaches the critical threshold, settling at an impressive **0.49 px** terminal error rate:

![Training Loss and Accuracy](https://github.com/RaphaelKha/ballistic-trajectory-forecasting/blob/main/fitting_loss_settings.png?raw=true)
*Left: The logarithmic descent of the Sniper Horizon Loss. Right: The terminal impact error converging below the sub-pixel threshold.*

---

### 5. Hardware Specifications & Edge Deployment (Artifacts)
To guarantee convergence and inference speed, the system was built with Edge Computing constraints in mind.

* **Hardware Hardware:** NVIDIA A100 (Ampere architecture).
* **Convergence Time:** < 5 minutes for 400 epochs, enabled by AdamW optimization and a fully vectorized DataLoader containing 20,000 synthetic trajectories.
* **The AI Artifact (`.pth`):** The final exported model weights contain only the learned synpatic `state_dict`. 
* **Storage Footprint:** **~1.5 MB**.

**Impact:** This Ultra-Lightweight memory footprint allows the model to be natively embedded on field micro-calculators (e.g., AA Turrets, Drones) yielding zero-latency, 1000+ FPS real-time inference without reliance on cloud compute.

---

### Conclusion
By embedding kinematic constraints directly into the loss landscape and utilizing an SSM/VideoMamba-inspired sequence architecture paired with an optimal assignment tracker, the system transcends simple pattern matching. It is robust to occlusions, fully translation-invariant, and highly scalable for real-time defense or acquisition systems.

---

### Copyright & License
© 2026 Raphael KHAYAT. All Rights Reserved.

*This repository and its documentation are provided for portfolio and demonstration purposes only. To protect intellectual property, the source code, training pipeline, and model weights are maintained in a separate private repository. A full code walkthrough and live demonstration can be provided upon request during technical interviews. No license is granted to use, copy, modify, or distribute this software.*
