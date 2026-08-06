# MPC Planner for Dataset Generation

<div align="center">
  <img src="https://img.shields.io/badge/Conference-IECON%202026-blueviolet" />
  <img src="https://img.shields.io/badge/Method-Model%20Predictive%20Control-blue" />
  <img src="https://img.shields.io/badge/Feature-Collision%20Avoidance-green" />
</div>

Deploy and visualize the MPC-based trajectory planner in paper "[Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition](https://arxiv.org/abs/2507.04384)". It can serve both as a high-quality dataset generator and as a real-time trajectory planner. 

<table>
  <tr>
    <td align="center" width="40%">
      <img src="./assets/teaser2.gif" alt="teaser" width="420" />
      <br/>
      <b>(a)</b> Only Consider Static Obstacles
    </td>
    <td align="center" width="40%">
      <img src="./assets/teaser1.gif" alt="teaser" width="420" />
      <br/>
      <b>(b)</b> Only Consider Dynamics Obstacles
    </td>
  </tr>
</table>

<br>

Jointly accounting for both obstacle types makes safe trajectory planning in unseen scenes particularly challenging. To address this issue, we employ an energy-parameterized diffusion composition approach that enables decision-making at inference time thus producing safe trajectories. For details, please refer to the [paper](https://arxiv.org/abs/2507.04384).

<div style="display: flex; justify-content: flex-start;">
	<img src="./assets/teaser.gif" alt="GIF 2" width="100%" style="margin-right: 10px;"/>
</div>
<br>

# 🪄 Quickstart

Start by cloning this repository to the host:

```bash
git clone https://github.com/zhouhengli/rstp-mpc-planner.git
```

## 📂 Folder Structure

```
.
├── assets
├── collision_check   # For posterior checking of collisions in rstp-mpc trajectories
├── config
├── dataset           # Reference trajectories generated using the ritp method
├── LICENSE
├── local_planner     # Core implementation of the MPC formulation
├── map
├── media
├── README.md
├── requirements.txt
└── scripts           # Entry point of the code
```

The [RITP](https://github.com/zhouhengli/ritp) method is a path-velocity decomposition planning approach characterized by exceptionally fast computation, achieving runtimes as low as 10 ms. In particular, the ItCA submodule in RITP effectively smooths discrete path points, thereby preserving trajectory feasibility.

## 🧮 Mathematical Formulas

The improved Euler method is used to discretize the kinematic model $f(\zeta),$ where $\zeta_0$ represents the current state of the vehicle. Then the optimization problem for the MPC planner across diverse scenes is expressed as follows:

```math
	\begin{equation}
		\begin{aligned}\label{eq:mpc}
			\min_{\boldsymbol{\zeta}, \mathbf{u}} \quad & \gamma \cdot J_\textrm{dynamic} +   \sum_{k=1}^{N_p}   \overbrace{ \left( \Vert \mathbf{A} \cdot \boldsymbol{\zeta}_k - \mathbf{p}_{s_k}^{\text{ref}} \Vert^2_{Q_1} \right)}^{J_\textrm{static}}  \\
			& + \sum_{k=1}^{N_p-1}\underbrace{ \Vert \Delta \mathbf{u}_k \Vert^2_{R_1}
				+ \Vert \mathbf{u}_k \Vert^2_{R_2}
				 + \Vert \mathbf{A} \cdot \boldsymbol{\zeta}_{N_p} - \mathbf{p}_{s_{N_p}}^{\text{ref}} \Vert^2_{Q_2}     }_{J_\textrm{static}} \\
			\text{s.t.} \quad
			& \boldsymbol{\zeta}_{k} = \boldsymbol{\zeta}_{k-1} + T_s \cdot f \left( \boldsymbol{\zeta}_{k-1} + \frac{T_s}{2} f \left( \boldsymbol{\zeta}_{k-1}, \mathbf{u}_k \right), \mathbf{u}_k \right), \\
			& \boldsymbol{\zeta}_0 = \boldsymbol{\zeta}_{\textrm{cur}}, \quad \boldsymbol{\zeta}_{\textrm{min}} \leq \boldsymbol{\zeta}_k \leq \boldsymbol{\zeta}_{\textrm{max}, \quad \mathbf{u}_{\textrm{min}} \leq \mathbf{u}_k \leq \mathbf{u}_{\textrm{max}}},
		\end{aligned}
	\end{equation}
```


where $\mathbf{A} = [1, 1, 1, 0]^{\top},$ and $N_p$ denotes the prediction horizon. $\zeta_{\text{cur}}$ is the current vehicle state. In dynamic scenes, $\gamma = 1$; otherwise, $\gamma = 0$.

## 🛠️ Configure

**[1/3] Create and Activate a Virtual Environment:** First, create a virtual environment using `conda` with Python 3.8, then activate it:

```bash
conda create -n rstp-mpc python=3.8
conda activate rstp-mpc
```

**[2/3] Install Dependencies:** Next, install all required dependencies using `pip` from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**[3/3] Run the Script:** Finally, you can run the script to initiate the process:

```bash
python scripts/mpc_planner.py
```

This will start the main functionality of the repository.

## 🤗 Acknowledgments

- [Vehicle_Motion_Planning_with_Obstacles_Avoidance_using_MPC](https://github.com/tg623623nana/Vehicle_Motion_Planning_with_Obstacles_Avoidance_using_MPC)
- [AutomatedValetParking](https://github.com/wenqing-2021/AutomatedValetParking)

Please contact [Zhouheng Li](https://zhouhengli.github.io/) if you have any questions or suggestions. A ⭐ would be greatly appreciated and would serve as strong encouragement for my continued open-source research efforts : )

## Citations

If you find this work useful, please consider starring this repository and citing the paper as follows:

```
@article{mao2025rapid, 
	title={Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition}, 
	author={Mao, Wule and Li, Zhouheng and Luo, Yunhao and Du, Yilun and Xie, Lei}, 
	journal={arXiv preprint arXiv:2507.04384}, 
	year={2025} 
}
```
