## Reference:
[Distributed Deep Learning-based Offloading for Mobile Edge Computing Networks](https://link.springer.com/article/10.1007/s11036-018-1177-x)

Huang, Liang, et al. "Distributed deep learning-based offloading for mobile edge computing networks." Mobile networks and applications 27.3 (2022): 1123-1130.

## Abstract

- Research Subject: Offloading computation tasks from wireless devices to nearby access points or base stations in mobile edge computing (MEC) networks.

- Objective: To develop a joint management strategy for offloading decisions and radio bandwidth allocation that minimizes system utility, defined as the weighted sum of energy consumption and task completion delay for all wireless devices (WDs).

- Challenges: The main challenge is managing binary offloading decisions and associated radio bandwidth allocation in real-time, given the computational complexity of enumerating all possible solutions and the trade-off between optimality and computational complexity.

- Solution Approach: Introduction of a distributed deep learning-based offloading algorithm. The framework uses multiple parallel deep neural networks (DNNs) to generate offloading decisions effectively and efficiently. These decisions are stored in shared memory to further train and improve the DNNs.

- Performance: The proposed algorithm achieves near-optimal offloading decisions in less than a second, converging to optimal performance when two or more DNNs are used. It demonstrates effectiveness under various parameter settings, making it suitable for real-time offloading in MEC networks.

## Key design
![Problem assumption](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Problem%20assumption.png)

![Optimizing structure](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Optimization%20structure.png)

![Model structure](https://github.com/Mpetof/MECN/blob/main/DROO/Figure/Model%20structure.png)