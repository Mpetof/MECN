## Reference:
[Deep Reinforcement Learning for Online Computation Offloading in Wireless Powered Mobile-Edge Computing Networks](https://ieeexplore.ieee.org/abstract/document/8771176/)

Huang, Liang, Suzhi Bi, and Ying-Jun Angela Zhang. "Deep reinforcement learning for online computation offloading in wireless powered mobile-edge computing networks." IEEE Transactions on Mobile Computing 19.11 (2019): 2581-2593.

## Abstract

- Research Subject: Wireless powered mobile-edge computing (MEC), specifically focusing on networks that utilize a binary offloading policy in low-power networks.

- Objective: To develop an online algorithm that optimally adapts task offloading decisions and wireless resource allocations to dynamic and time-varying wireless channel conditions.

- Challenges: The main challenge lies in the need to quickly solve hard combinatorial optimization problems within the channel coherence time, which traditional numerical optimization methods struggle to achieve efficiently.

- Solution Approach: Introduction of the Deep Reinforcement learning-based Online Offloading (DROO) framework. This framework uses a deep neural network to learn binary offloading decisions from past experiences, bypassing the need to directly solve combinatorial optimization problems and thereby reducing computational complexity.

- Performance: The DROO framework not only reduces the computation time significantly (by more than an order of magnitude compared to existing methods) but also achieves near-optimal performance. It delivers extremely low CPU execution latencies (under 0.1 seconds in a network with 30 users), thus enabling real-time, optimal offloading even in fast fading environments.

## Key design
![Problem assumption](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Problem%20assumption.png)

![Optimizing structure](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Optimization%20structure.png)

![Model structure](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Model%20structure.png)

## Remark
