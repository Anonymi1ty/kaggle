# discussion汇总

基本方法[概述](https://www.kaggle.com/competitions/lux-ai-season-3/discussion/556056)

本次可以参考的agent相关[比赛参考](https://www.kaggle.com/competitions/lux-ai-season-3/discussion/550848)

agent[策略](https://www.kaggle.com/competitions/lux-ai-season-3/discussion/551158)

（看到relic node point tile location）



## Code

**Relicbound (most voted)**是一个好的起步notebook，但是只是汇总了简单的寻路策略

[NuralBrain v0.5 model | Train And Win](https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-model-train-and-win/notebook) DQN 强化学习框架 ，RL-agent的应用（但是仍然是手动进行对抗决策，单一agent进行训练）==改进训练方法就可以铜牌==

## 一些思路

>#### 行动顺序
>
>在比赛的每个时间步，我们按以下顺序执行以下步骤：
>
>1. 移动所有有足够能量进行移动的单位。
>2. 执行所有有足够能量进行汲取动作的单位的汲取操作。
>3. 处理碰撞并应用能量空洞场的影响。
>4. 根据单位所在位置（如能量场和星云地块）更新所有单位的能量。
>5. **为所有团队生成新单位**，并移除能量小于 0 的单位。
>6. 确定所有团队的视野/传感器掩码，并据此屏蔽观察结果。
>7. 让环境中的物体（如小行星、星云地块、能量节点）在空间中移动。
>8. 计算新的团队积分。
>
>需要注意的是，每场比赛运行 `params.max_steps_in_match` 步，您可以执行相应数量的动作来影响游戏。然而，您实际上会收到 `params.max_steps_in_match + 1` 帧的观察结果，因为第一帧要么是空的，要么是上一场比赛的最终观察结果（基于这些观察结果执行的动作不会产生任何效果）。

首先，==因为每一回合都会生成一个新的单位，所以是一个muti-agnet的过程（需要不同agent的配合）==



## 方向

### 经典的Centralized  muti-agent算法

PRIMAL，MADDPG，GDQ，

#### MAPPO（2022，将基础的单agent算法扩展到muti-agent上，==推荐，业界标杆star高，代码多==）

1. MAPPO（Multi-Agent Proximal Policy Optimization）之所以被归类为**Centralized Multi-Agent Reinforcement Learning (MARL)** 算法，主要是因为它在训练阶段采用了**集中式的训练**框架，同时在执行阶段则是**分布式的策略执行**。这种“集中式训练，分布式执行”（Centralized Training with Decentralized Execution, CTDE）的设计是MARL领域常见的范式。

   以下详细解释为什么MAPPO被视为一种**Centralized Multi-Agent**算法：

   ------

   ### 1. **集中式训练的核心思想**

   在多智能体场景下，每个智能体的策略优化可能需要知道其他智能体的状态或动作信息，以便在复杂环境中更好地协调和协作。然而，在实际执行过程中，智能体通常只能根据自己的局部信息（partial observation）做决策。

   MAPPO通过以下方式实现集中式训练：

   1. 访问全局信息

      ：在训练阶段，每个智能体可以访问**全局状态（global state）**和其他智能体的观测或动作，这些信息有助于更准确地估计价值函数或优势函数。

      - 例如：价值函数 V(s)V(s) 可以根据整个环境的全局状态 ss 来估计，而不仅仅依赖于个体的局部观测。

   2. **共享经验**：所有智能体的经验可以集中起来，共享用于训练目标函数。这种集中化方式能更高效地利用数据。

   由于训练阶段可以获取所有智能体的全局信息和联合策略表现，MAPPO具备了集中式的特性。

   ------

   ### 2. **去中心化的策略执行**

   虽然训练过程中使用了全局信息，但在执行阶段，MAPPO的策略是去中心化的。这意味着：

   - 每个智能体仅根据自身的局部观测$o_i$ 和训练好的策略 $\pi(a_i | o_i) $做出决策。
   - 不需要全局状态或其他智能体的信息，保证了执行的分布式特性。

   这种设计使得MAPPO可以应用于现实场景中，例如机器人团队协作或分布式控制系统，在执行时没有全局信息的依赖。

   ------

   ### 3. **Centralized Training with Decentralized Execution (CTDE)**

   MAPPO遵循了CTDE的范式，这是许多多智能体算法（如MADDPG、QMIX等）的核心思想：

   1. **训练阶段**：每个智能体的策略更新是基于全局信息（如全局状态、全局奖励或其他智能体的行为）的。这种集中式的训练能够更高效地优化智能体之间的协作。
   2. **执行阶段**：每个智能体只需要使用自身的局部观测，策略是完全去中心化的。

   这种范式的好处是：

   - 集中式训练提高了策略优化的效率。
   - 去中心化执行保证了算法的可扩展性和现实可行性。

   ------

   ### 4. **MAPPO的具体设计**

   #### **价值函数的集中式估计**

   MAPPO通常使用一个集中式的价值函数 V(s)，其中 s 是全局状态：

   $L^{VF} = \frac{1}{2} \mathbb{E} \left[ \left( V(s) - V_\text{target} \right)^2 \right]$

   这个集中式的值函数能更准确地估计全局状态下的价值，而不仅仅依赖于个体的局部观测。

   #### **共享策略与独立决策**

   - **策略共享**：在多智能体系统中，MAPPO可以为所有智能体共享一个统一的策略，或者为每个智能体训练独立的策略。
   - **局部观测决策**：在执行阶段，智能体基于自身的局部信息 $o_i $来独立决策，避免了对全局信息的依赖。

   ------

   ### 5. **与去中心化算法的区别**

   相比完全去中心化的多智能体算法（如Independent PPO, IPPO），MAPPO显著的不同在于：

   1. **集中式的值函数估计**：MAPPO允许值函数使用全局状态进行估计，而IPPO的值函数只能依赖于个体的局部信息。
   2. **更强的协作能力**：通过全局信息的共享和集中式训练，MAPPO能更好地优化多个智能体之间的协作和协调。

   这种集中式设计使得MAPPO在需要强协作的场景（例如合作博弈、多机器人任务）中表现出更高的性能。

   ------

   ### 6. **为什么MAPPO被算作Centralized Multi-Agent算法？**

   总结来说，MAPPO被视为集中式多智能体算法的原因在于：

   - **集中式信息使用**：在训练阶段，MAPPO利用了全局状态、全局奖励或其他智能体信息进行策略优化。
   - **集中式值函数估计**：通过全局状态 $s $来优化价值函数，增强了智能体间的协作能力。
   - **CTDE范式**：MAPPO典型地遵循了集中式训练与去中心化执行的框架，这是Centralized Multi-Agent算法的重要特征。

https://paperswithcode.com/paper/the-surprising-effectiveness-of-mappo-in

### 经典的Decentralized  muti-agent算法

> 检查agent的运动策略，因为要和比赛规则相同（只能↑、→、↓、←、*）
>
> 并且一把游戏有100个steps，所以max-agent最好要在100以上

#### ~~MAPPER （2020）~~

>MAPPER 为代理添加了四个额外的对角线移动（↗、↘、↙、↖），Max-agent：150==没代码==

#### G2RL （2020）

> Max-agent：128

使用DDQN和A*规划器，优点是由于代理之间不需要通信，因此该方法不需要考虑通信延迟，并且可以扩展到任何规模。

https://github.com/Tushar-ml/G2RL-Path-Planning（github issues：有人反馈和论文结果差别较大）

#### PRIMAL2（2021）

>Max-agent：2048

离散方法，够用了

https://github.com/marmotlab/PRIMAL2

#### ~~PIPO（2022）~~

> Max-agent：512==没代码==