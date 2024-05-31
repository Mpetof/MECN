## Reference:
[Lyapunov-guided deep reinforcement learning for stable online computation offloading in mobile-edge computing networks](https://ieeexplore.ieee.org/abstract/document/9449944/)

Bi, Suzhi, et al. "Lyapunov-guided deep reinforcement learning for stable online computation offloading in mobile-edge computing networks." IEEE Transactions on Wireless Communications 20.11 (2021): 7519-7537.

## Abstract

- Research Subject: Opportunistic computation offloading in multi-user mobile-edge computing (MEC) networks with time-varying wireless channels and stochastic user task data arrivals.

- Objective: To design an online computation offloading algorithm that maximizes network data processing capability while ensuring long-term data queue stability and average power constraints.

- Challenges: The main challenge is making decisions in each time frame without knowledge of future random channel conditions and data arrivals, addressing the complexity of multi-stage stochastic mixed integer non-linear programming (MINLP) problems.

- Solution Approach: Introduction of the LyDROO framework, which combines Lyapunov optimization and deep reinforcement learning (DRL). Lyapunov optimization decouples the multi-stage stochastic MINLP into deterministic per-frame MINLP subproblems, ensuring all long-term constraints are met. DRL then solves these per-frame subproblems with low computational complexity.

- Performance: LyDROO achieves optimal computation performance, stabilizes all system queues, and induces very low computation time, making it suitable for real-time implementation in fast fading environments. Simulation results validate its effectiveness under various network setups.

## Key design
![Problem assumption](https://github.com/Mpetof/MECN/blob/main/LyDROO/Figure/System%20mode.png)

![Optimizing structure](https://github.com/Mpetof/MECN/blob/main/LyDROO/Figure/Model%20structure.png)
