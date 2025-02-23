# PPO复现

## 核心问题

同一个代码，如果在调用main函数时返回任何任务台报错或者警告，将会影响agent的运行结果（官方文档说，print会被视为agent的输出）：

![image-20250222190247079](https://my-typora-p1.oss-cn-beijing.aliyuncs.com/typoraImgs/image-20250222190247079.png)

![image-20250222190310664](https://my-typora-p1.oss-cn-beijing.aliyuncs.com/typoraImgs/image-20250222190310664.png)

1.对于我们原本环境，可以改为和他们一样的四楼，如果load函数报错，那么agent.py文件可以直接作为主函数不写进去

2.手写开发环境多此一举，`from luxai_s3.wrappers import LuxAIS3GymEnv`直接调比赛环境。

## 在V4版本上进行的修改

==1.在训练环境下模拟比赛数据传输格式：==

```json
// T is the number of teams (default is 2)
// N is the max number of units per team
// W, H are the width and height of the map
// R is the max number of relic nodes
{
  "obs": {
    "units": {
      "position": Array(T, N, 2),
      "energy": Array(T, N, 1)
    },
    // whether the unit exists and is visible to you. units_mask[t][i] is whether team t's unit i can be seen and exists.
    "units_mask": Array(T, N),
    // whether the tile is visible to the unit for that team
    "sensor_mask": Array(W, H),
    "map_features": {
        // amount of energy on the tile
        "energy": Array(W, H),
        // type of the tile. 0 is empty, 1 is a nebula tile, 2 is asteroid
        "tile_type": Array(W, H)
    },
    // whether the relic node exists and is visible to you.
    "relic_nodes_mask": Array(R),
    // position of the relic nodes.
    "relic_nodes": Array(R, 2),
    // points scored by each team in the current match
    "team_points": Array(T),
    // number of wins each team has in the current game/episode
    "team_wins": Array(T),
    // number of steps taken in the current game/episode
    "steps": int,
    // number of steps taken in the current match
    "match_steps": int
  },
  // number of steps taken in the current game/episode
  "remainingOverageTime": int, // total amount of time your bot can use whenever it exceeds 2s in a turn
  "player": str, // your player id
  "info": {
    "env_cfg": dict // some of the game's visible parameters
  }
}
```

==2.逐步增加agent的个数也要改（按照比赛规则）==

==3.计算视野（根据不同agent id进行计算）==

比赛中json是返回一队的视野观察，所以想要根据不同unit计算不同视野，产生不同的预测，架构如下：

>按照这个奖励函数修改

![image-20250211233804718](https://my-typora-p1.oss-cn-beijing.aliyuncs.com/typoraImgs/image-20250211233804718.png)

==4.奖励函数优化==

```python
    """
    2. 奖励函数优化：
       - 对每个己方单位分别计算奖励后求和。奖励包括：
         - 检查移动是否越界或移动到 Asteroid Tiles，若是则扣 -4.0
         - Sap 动作：如果单位可见至少一个 relic，则统计其 8 邻域内敌方单位数，
           若数量>=2，则 sap 奖励 = +1.0 × 敌方单位数，否则扣 -4.0；
         - 非 sap 动作：检查单位所在位置是否处于 relic 配置内（即潜力点），
           若第一次访问奖励 +2.0，若该潜力点产生 team point（score 增加）则额外奖励 +5.0，
           并且只要停留在该点，每回合奖励 +5.0；同时能量节点奖励+0.2，Nebula 惩罚 -1.0；
         - 攻击行为：如果单位移动后与敌方单位重合且敌方能量低于己方，则奖励 +1.0 每个敌人；
         - 全局探索奖励：每新发现一个 tile 奖励 +0.05。
    """
```



==5.把原有的模型加载进steps步骤作为对抗组训练（待完成）==

==6.能量管理与消耗（待完成）==

​	**能量未更新**
在真实比赛规则中，单位的移动和汲取动作都会消耗能量，且单位能量会因处于不同地形（如星云或能量节点）而发生变化。但在你提供的 PPOGameEnv 中：

- 单位的“energy”属性在初始时设为 100，但在 step() 中并没有对能量进行扣减或补充（例如移动消耗、汲取消耗、碰撞时能量变化等）。
- 这会导致训练过程中各单位的能量一直保持不变，进而使得模型无法学到在能量管理方面做出决策（例如在能量低时避免激进的动作）。



### 存在的问题

最后的训练结果存在和奖励函数设计不符的情况。

这应该是该notebook的最后一个版本，它是一个很好的起步框架（虽然它的结果需要优化）





### debug 流程

1.`self.observation_space = None`语句和整体环境不匹配

需要定义一个类似于上面json格式的字典

2.字典不能嵌套处理，需要进行flat

3.字典中spaces.Text()命名空间错误，删除该行

4.使用Dict的observation_space在train中需要将参数和改为`MultiInputPolicy`

5.初始化问题，只修改`self.observation_space`不合理，同时也要修改和其联动函数，比如`get_obs()` 方法和`get_unit_obs`，以及 `reset()`

6.因为对齐问题，现在`get_obs()` 和 `observation_space` 完全匹配（进行了flat处理）其他部分比如`get_unit_obs`仍然使用json格式



### 参数对齐问题

传入的**obs**的形式

```json
output = {
    'units': {
        'position': np.array([
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
             [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1],
             [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
        ]),
        'energy': np.array([
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        ])
    },
    'units_mask': np.array([
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    ]),
    'sensor_mask': np.array([
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26,
        [False] * 26
    ]),
    'map_features': {
        'energy': np.full((24, 24), -1),
        'tile_type': np.full((24, 24), -1)
    },
    'relic_nodes': np.array([
        [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]
    ]),
    'relic_nodes_mask': np.array([False, False, False, False, False, False]),
    'team_points': np.array([0, 0]),
    'team_wins': np.array([0, 0]),
    'steps': 0,
    'match_steps': 0
}

```

**remainingOverageTime**键缺失



**units_energy**返回只有2维度,虽然强行给他改成正确的了（加了一维度，但是返回I信息不一定对）

```

units_energy raw: [[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1][-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]]
units_energy shape before: (2, 16)
```



如果 unit 当前所在位置显示遗迹且没有其他友方单位或者敌方单位未占领,移动到遗迹上面奖励 +3.0     and, 如果发现自己的位置已经在遗迹上，若采用 center（保持不动）动作，奖励 +5.0 对于任何unit： 若己方自身处于能量节点位置，reward += 1.0 若处于 Nebula（tile_map==1），reward -= 1.0

### 训练结果

问题：

>1. 在使用`!luxai-s3 agent/main.py agent/main.py --output=replay.html`指令生成模拟对抗的时候，agent不会主动去get points（占领遗迹点 
>
>2. 没有判断移动是否可以，具体示例是在地图边界或者在Asteroid Tiles旁边仍然往不能移动的方向移动。 
>3. 一些单位平白的使用sap动作，平白无故消耗自己的能量及时他们周围没有敌方单位

修改：

还有！！看看规则，哪个地块加分

~~~markdown
对于1.gent不会主动去get points的问题，我发现了主要原因：
```
我的模拟环境和游戏运行的真实环境不匹配，具体在get points方式上和游戏的结算上。
在get points方式上：游戏介绍中说了`在代码中，**会为每个遗物节点生成一个以遗物节点为中心的随机 5x5 configuration /掩码，用于指示哪些地块可以产生积分（一般一个遗迹周围有5个地块能产生积分）**，哪些不能。能够产生积分的地块始终是隐藏的，只有通过反复尝试在遗物节点周围移动才能发现。遗物节点本身的位置可以通过传感器范围内的观察发现。`.但是现在的奖励机制是占领遗迹节点即可获得reward，这个和原本的游戏是不一样的。
在游戏的结算方面，对于这些占领了configuration 指示的地块的队伍，每占领一个这样的地块的team_points需要每回合+1。每一轮游戏进行100个step，然后恢复每个team到初始状态，重置configuration，其他东西不变。每5轮游戏（500step）就要全部重新生成一个新的地图。
所以需要你对ppo_game_env.py进行修改。

对于奖励函数进行的修改方面，下面是可能的一些逻辑，请你参考并给出一个完善的版本：
对于每一个unit，分别计算他们不同的unit_reward；
对于每一个agent的动作都要判断是否会超出SPACE_SIZE*SPACE_SIZE的地图边界或者下一步到达的位置是否是Asteroid Tiles类型，如果有，该动作被判定为无效动作，并且unit_reward-4.0
- Sap 动作：基于 get_unit_obs 中的信息，判断是否有遗迹区域；如果有，检查遗迹周围5*5的范围内是否有敌方单位；如果有敌方单位，向敌方单位移动，并尽可能让四邻域内（8个space）有尽可能多的敌方单位；如果四邻域内（8个space）敌方单位的数目大于等于2，则使用sap动作，并奖励+(1.0*敌方单位的数目),否则其他情况使用sap动作都视为惩罚unit_reward-4.0； 
- 非 Sap 动作：，
            1.基于己方所有单位联合视野中新发现了一个relic，则根据这个relic的position为中心，设置一个5*5的potential_points_space的矩阵，unit访问这个矩阵的一个没有visited的space就unit_reward+2.0；如果team_points没有因为占领这一个space多加1分，则将这个space设置为visited；如果因为占领了这一个space导致team_points多加了1分，则再unit_reward+5.0，并且标记该点为team_points_space。并且只要呆在这个space上每回合unit_reward+5.0。
            2.对于任何unit： 若己方自身处于能量节点位置，unit_reward+= 0.2
            3. 若处于 Nebula（tile_map==1），unit_reward-= 1.0
4.更新全局探索奖励：基于己方所有单位联合视野中新发现的 tile（从-1变为了其他类型），total_reward+(0.05 * 新探索的space)
5.如果在己方所有单位联合视野中新发现relic，那么除了占领team_points_space的unit和正在探查potential_points_space的unit，其他的unit都应该向potential_points_space靠拢，并且探查potential_points_space。（没有设计奖励细则，请自行补充）
6.攻击行为，如果检测到敌方单位，并且该单位能量值低于unit自身，则向敌方单位移动并且覆盖敌方单位（攻击行为），完成该动作unit_reward+= 1.0
```
需要你对代码进行修改。
~~~



先检查推理环境是否正确

==针对2.修改：warp函数（环绕边界检查）直接移除并进行修改；增加出界和碰到小型星的处罚力度-5.0==

==针对3.修改：加大如果周围没有地方单位的使用sap出处罚力度-5.0==



​	

再确定检查环境问题，每个信息是否获取到



现在我在着重分析train部分可能的问题，请你仔细阅读上面代码，我待会会给你一些训练结果，请结合

再检查棋子走路逻辑，position的逻辑是否和我们训练中定义的对上





# MAPPO

### ENV

> debug可以按照chat方法打印

1.查看接口传参规范，对齐比赛数据和训练部分的数据

2.设计自己需要的函数，比如想要实现什么逻辑（奖励函数等等）？

### agent

定义agent行为和相关奖励函数



### train

1.加载环境直接用`from luxai_s3.wrappers import LuxAIS3GymEnv`

2.初始化两个agent对抗训练



（对手的多样性，设计类似经验放回池子checkpoints,）

直接开始训练

### 运行

main.py

submission.tar.gz

