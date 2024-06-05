## Reference:
[Offloading and Resource Allocation With General Task Graph in Mobile Edge Computing: A Deep Reinforcement Learning Approach](https://ieeexplore.ieee.org/abstract/document/9093962/)

Yan, Jia, Suzhi Bi, and Ying Jun Angela Zhang. "Offloading and resource allocation with general task graph in mobile edge computing: A deep reinforcement learning approach." IEEE Transactions on Wireless Communications 19.8 (2020): 5404-5419.

## Abstract

- Research Subject: Mobile-edge computing (MEC) system with one Access Point (AP) and one Mobile Device (MD) 

- Objective: Optimize task offloading decisions and resource allocation (e.g., CPU computing power) to minimize the Energy-time cost (ETC) for the MD. 

- Challenges: Complexity of combinatorial offloading choices and the intricate interdependencies of task executions under the general dependency model, especially for large problem sizes. 

- Solution Approach: Propose a DRL framework leveraging an actor-critic learning structure, where the actor network for determining the optimal binary offloading decisions and the critic network for low complexity assessing ETC. Utilize an "oneclimb" structure within the optimal offloading decision, narrowing down the action search space. 

- Performance: Numerical results indicate that the proposed algorithm achieves up to 99.1% of the optimal performance while markedly reducing computational complexity compared to traditional optimization methods

## Key design
![Problem assumption](https://github.com/Mpetof/MECN/blob/main/GeneralTaskGraph/Figure/Task%20graph.png)

![Optimizing structure](https://github.com/Mpetof/MECN/blob/main/GeneralTaskGraph/Figure/Model%20structure.png)
