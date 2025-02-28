import numpy as np
import copy
import gym
from luxai_s3.wrappers import LuxAIS3GymEnv

class HarlLuxAIS3Env:
    def __init__(self, seed=42, player_id="player_0"):
        self.seed_val = seed
        self.current_step = 0
        # 实例化原环境（可根据需要选择 numpy_output 参数）
        self.env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = self.env.reset(seed=seed)
        env_cfg = info["params"]
        
         # 保存 player_id 及友/敌索引
        self.player_id = player_id
        if player_id == 'player_0':
            self.friendly_idx, self.enemy_idx = 0, 1
            self.enemy_id = 'player_1'
        else:
            self.friendly_idx, self.enemy_idx = 1, 0
            self.enemy_id = 'player_0'
        
        # 存储游戏相关参数
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.max_relic_nodes = info["full_params"]["max_relic_nodes"]
        self.max_point_nodes = 6
        self.local_window_size = env_cfg["unit_sensor_range"] * 2 + 1
        
        # 假设每队最多 16 个单位，只考虑己方
        self.max_units = env_cfg["max_units"]
        self.n_agents = self.max_units
        
        # 原环境动作空间为每队一个 (max_units, 3) 的 Box，
        # 这里拆分为每个 agent 一个 (3,) 的 Box（取值范围 0~5）
        self.action_space = [gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.int16)
                             for _ in range(self.n_agents)]
        
        # 只构造己方的共享观测空间
        team_obs = obs[self.player_id]
        self.share_observation_space = [copy.deepcopy(team_obs) for _ in range(self.max_units)]

        
        # 初始化 potential_point 矩阵，形状为 (max_relic_nodes, 5, 5),默认初始值为 -1
        self.potential_point = [[[-1 for _ in range(5)] for _ in range(5)]
                   for _ in range(self.max_relic_nodes)]
        
        # 构造每个 agent 的观测空间（只使用己方观测）
        self.observation_space = []
        for unit_id in range(env_cfg["max_units"]):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(team_obs, player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size, self.potential_point)
            self.observation_space.append(unit_obs)
    
    @staticmethod
    def construct_unit_obs(player_obs, player_id, unit_id,
                             unit_sap_range, unit_move_cost, unit_sap_cost,
                             max_relic_nodes, max_point_nodes,
                             local_window_size=5, potential_point = None):
        """
        构造当前单位的局部观测信息

        参数：
          player_obs: 当前玩家的观测字典（如 obs[player_id]）
          player_id: 当前玩家的 id，例如 "player_0" 或 "player_1"
          unit_id: 需要构造观测的单位在己方列表中的索引
          其它参数为游戏相关参数
          local_window_size: 用于提取局部地图信息的窗口大小（默认 5x5）

        返回：
          包含 unit_obs 信息的字典
        """
        # 判断友方和敌方的索引（底层观测依然包含两队信息）
        if player_id == 'player_0':
            friendly_idx, enemy_idx = 0, 1
        else:
            friendly_idx, enemy_idx = 1, 0

        # 从全局观测中提取友方和敌方单位信息
        friendly_units = []
        enemy_units = []
        N = len(player_obs['units_mask'][friendly_idx])
        for i in range(N):
            if player_obs['units_mask'][friendly_idx][i]:
                friendly_units.append({
                    'position': player_obs['units']['position'][friendly_idx][i],
                    'energy': player_obs['units']['energy'][friendly_idx][i]
                })
            # 如果是当前 unit_id，但是该单位已丢失，则默认值
            else:
                friendly_units.append({
                    'position': [-1, -1],
                    'energy': -1
                })
        for i in range(N):
            if player_obs['units_mask'][enemy_idx][i]:
                enemy_units.append({
                    'position': player_obs['units']['position'][enemy_idx][i],
                    'energy': player_obs['units']['energy'][enemy_idx][i]
                })
            # 如果是当前 unit_id，但是该单位已丢失，则默认值
            else:
                friendly_units.append({
                    'position': [-1, -1],
                    'energy': -1
                })

        # 提取当前单位的状态（假设 unit_id 合法且该单位存在）
        self_state = friendly_units[unit_id]

        # 提取局部地图信息：以单位当前位置为中心，获取 window_size x window_size 的 patch
        def get_local_map(unit_pos, map_features, sensor_mask, window_size):
            x, y = int(unit_pos[0]), int(unit_pos[1])
            half = window_size // 2
            W = len(map_features['energy'])
            H = len(map_features['energy'][0])
            local_energy = []
            local_tile_type = []
            for i in range(x - half, x + half + 1):
                row_energy = []
                row_tile = []
                for j in range(y - half, y + half + 1):
                    if 0 <= i < W and 0 <= j < H:
                        row_energy.append(map_features['energy'][i][j])
                        row_tile.append(map_features['tile_type'][i][j])
                    else:
                        row_energy.append(0)
                        row_tile.append(-1)
                local_energy.append(row_energy)
                local_tile_type.append(row_tile)
            return {
                'local_energy': local_energy,
                'local_tile_type': local_tile_type
            }

        local_map = get_local_map(self_state['position'],
                                  player_obs['map_features'],
                                  player_obs['sensor_mask'],
                                  local_window_size)

        # 计算可见的 relic_nodes
        avail_relic_nodes = []
        R = len(player_obs['relic_nodes_mask'])
        for i in range(R):
            if player_obs['relic_nodes_mask'][i]:
                avail_relic_nodes.append(player_obs['relic_nodes'][i])

        # 敌方单位信息：判断是否在 sensor_mask 内可见
        visible_enemies = []
        def is_visible(pos, sensor_mask):
            x, y = int(pos[0]), int(pos[1])
            W = len(sensor_mask)
            H = len(sensor_mask[0])
            if 0 <= x < W and 0 <= y < H:
                return sensor_mask[x][y]
            return False

        for enemy in enemy_units:
            if is_visible(enemy['position'], player_obs['sensor_mask']):
                visible_enemies.append(enemy)

        unit_obs = {
            "self_state": self_state,             # 自身状态信息
            "local_map": local_map,                # 局部地图 patch 信息
            "visible_enemies": visible_enemies,    # 可见的敌方单位列表
            "avail_relic_nodes": avail_relic_nodes,# 可见遗迹节点位置列表
            "team_points": player_obs['team_points'],  # 比赛积分信息
            "params": {
                "unit_sap_range": unit_sap_range,
                "unit_move_cost": unit_move_cost,
                "unit_sap_cost": unit_sap_cost,
                "max_relic_nodes": max_relic_nodes,
                "max_point_nodes": max_point_nodes
            },
            "potential_point": potential_point     # 初始化的 potential point 矩阵
        }

        return unit_obs

    def reset(self, seed=None):
        """
        重置环境，返回 agent 级别观测、共享观测、infos 和可用动作。
        """
        self.current_step = 0
        
        # 初始化 potential_point 矩阵，形状为 (max_relic_nodes, 5, 5),默认初始值为 -1
        self.potential_point = [[[-1 for _ in range(5)] for _ in range(5)]
                   for _ in range(self.max_relic_nodes)]
        if seed is None:
            seed = self.seed_val + 1
        
        obs, info = self.env.reset(seed=seed)
        team_obs = obs[self.player_id]
        enemy_obs = obs[self.enemy_id]
        
        obs_agents = []
        enemy_obs_return = []
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(team_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size, self.potential_point)
            obs_agents.append(unit_obs)
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(enemy_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size, self.potential_point)
            enemy_obs_return.append(unit_obs)
        
        s_obs = [copy.deepcopy(team_obs) for _ in range(self.max_units)]
        available_actions = self.action_space.copy()
        infos = [info] * self.n_agents
        
        # 保存初始的 team_points 用于后续比较
        self.prev_team_points = team_obs['team_points']
        self.delta = 0
        return obs_agents, s_obs, available_actions, enemy_obs_return, infos

    #TODO:检查chat给的内容
    def get_unit_reward(self, unit_id, team_obs, obs_agents, s_obs, action_info):
        """
        计算单个 unit 的奖励。

        参数：
        unit_id: 单位索引（0~max_units-1）
        team_obs: 己方全局观测，包含 team_points, team_wins, relic_nodes, relic_nodes_mask, sensor_mask, units 等信息
        obs_agents: 各 agent 的局部观测列表（unit_obs 格式，其中 self_state 包含 energy 和 position）
        s_obs: 共享观测列表（通常为 team_obs 的深拷贝）
        action_info: 字典，记录该 unit 在本步执行动作时的额外信息，如 sap_success、sap_fail、illegal_move 等

        返回：
        单个 unit 的总奖励（float）。
        """
        reward = 0.0

        # 1. 终局奖励（Zero-Sum Terminal Reward）
        # 每 101 步检查 team_wins 是否有变化
        if self.current_step % 101 == 0:
            if team_obs['team_wins'] > self.prev_team_wins:
                reward += 2.0
            else:
                reward -= 2.0

        # 2. 中间奖励塑形

        ## 能量采集奖励：对比本步与上步单位能量差
        curr_energy = obs_agents[unit_id]["self_state"]["energy"]
        prev_energy = self.prev_unit_energy[unit_id] if hasattr(self, "prev_unit_energy") else curr_energy
        energy_gain = curr_energy - prev_energy
        if energy_gain > 0:
            reward += (energy_gain / 10.0) * 0.01  # 每增加10点能量奖励 +0.01

        ## 位置优势奖励：检查 unit 相对于每个可见 relic node 的位置
        # 这里的判断与 potential_point 更新逻辑保持一致
        for relic_idx, relic in enumerate(team_obs['relic_nodes']):
            if team_obs['relic_nodes_mask'][relic_idx]:
                unit_pos = obs_agents[unit_id]["self_state"]["position"]
                dx = int(unit_pos[0]) - int(relic[0])
                dy = int(unit_pos[1]) - int(relic[1])
                if -2 <= dx <= 2 and -2 <= dy <= 2:
                    idx_x = dx + 2
                    idx_y = dy + 2
                    if self.potential_point[relic_idx][idx_x][idx_y] == 1:
                        reward += 1.0
                    else:
                        reward += 0.2

        ## 冲突/冲突奖励：如果在当前 tile 同时存在我方和敌方单位，
        ## 且该 tile 上我方单位总能量高于敌方，则给予奖励
        # 根据 player_id 推断友/敌索引
        if self.player_id == 'player_0':
            friendly_idx, enemy_idx = 0, 1
        else:
            friendly_idx, enemy_idx = 1, 0
        unit_pos = obs_agents[unit_id]["self_state"]["position"]
        tile_x, tile_y = int(unit_pos[0]), int(unit_pos[1])
        friendly_sum = 0
        enemy_sum = 0
        # 遍历友方单位
        for i, mask in enumerate(team_obs['units_mask'][friendly_idx]):
            if mask:
                pos = team_obs['units']['position'][friendly_idx][i]
                if int(pos[0]) == tile_x and int(pos[1]) == tile_y:
                    friendly_sum += team_obs['units']['energy'][friendly_idx][i]
        # 遍历敌方单位
        for i, mask in enumerate(team_obs['units_mask'][enemy_idx]):
            if mask:
                pos = team_obs['units']['position'][enemy_idx][i]
                if int(pos[0]) == tile_x and int(pos[1]) == tile_y:
                    enemy_sum += team_obs['units']['energy'][enemy_idx][i]
        # 若冲突发生且我方能量更高（且至少有敌方单位存在）
        if enemy_sum > 0 and friendly_sum > enemy_sum:
            reward += 0.5

        ## 成功汲取/非法操作奖励：使用 action_info 进行判断
        if action_info.get("sap_success", False):
            reward += 0.5
        if action_info.get("sap_fail", False):
            reward -= 0.3
        if action_info.get("illegal_move", False):
            reward -= 0.5

        ## 视野扩展奖励：根据 sensor_mask 的增量给予少量奖励
        current_sensor = np.sum(team_obs['sensor_mask'])
        prev_sensor = self.prev_sensor_count if hasattr(self, "prev_sensor_count") else current_sensor
        sensor_diff = current_sensor - prev_sensor
        reward += sensor_diff * 0.003

        ## 单位损失（如果 unit 已丢失则重罚）
        if obs_agents[unit_id]["self_state"] is None:
            reward -= 2.0

        return reward


    #TODO:检查chat给的内容
    def step(self, actions, enemy_actions = None):
        """
        执行一步环境交互。
        参数 actions 为长度为 n_agents (16) 的动作列表, actions = [n_agents*action]
        返回：obs_agents, 共享观测, rewards, dones, infos, available_actions
        """
        # 对己方单位动作（长度为16）进行处理
        actions_team = np.stack(actions)  # shape: (max_units, 3)
        # 对于敌方，填充默认空动作（例如全 0 向量）
        dummy_enemy_actions = np.zeros((self.max_units, 3), dtype=np.int16)
        
        # 构造动作字典：只考虑己方，敌方统一填充空动作
        if enemy_actions is not None:
            if self.friendly_idx == 0:
                action_dict = {"player_0": actions_team, "player_1": enemy_actions}
            else:
                action_dict = {"player_0": enemy_actions, "player_1": actions_team}
        else:       
            if self.friendly_idx == 0:
                action_dict = {"player_0": actions_team, "player_1": dummy_enemy_actions}
            else:
                action_dict = {"player_0": dummy_enemy_actions, "player_1": actions_team}
        
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        team_obs = obs[self.player_id]
        enemy_obs = obs[self.enemy_id]
        
        # 构建己方和敌方的局部观测
        obs_agents = []
        enemy_obs_return = []
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(team_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size, self.potential_point)
            obs_agents.append(unit_obs)
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(enemy_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size, self.potential_point)
            enemy_obs_return.append(unit_obs)
        
        # -------------------------------
        # 更新 potential_point 的逻辑（unit级别）
        new_team_points = team_obs['team_points']
        new_delta = new_team_points - self.prev_team_points  # 本步 team_points 增速
        
        # 从 team_obs 中提取己方单位列表
        friendly_units = []
        N = len(team_obs['units_mask'][self.friendly_idx])
        for i in range(N):
            if team_obs['units_mask'][self.friendly_idx][i]:
                friendly_units.append({
                    'position': team_obs['units']['position'][self.friendly_idx][i],
                    'energy': team_obs['units']['energy'][self.friendly_idx][i]
                })
        
        # 对每个可见的 relic node（通过 relic_nodes_mask 判断）
        for relic_idx in range(len(team_obs['relic_nodes_mask'])):
            if team_obs['relic_nodes_mask'][relic_idx]:
                # 初始化该 relic node 的 potential_point 为全 0 的 5x5 矩阵
                self.potential_point[relic_idx] = [[0 for _ in range(5)] for _ in range(5)]
                relic_pos = team_obs['relic_nodes'][relic_idx]
                # 遍历每个己方 unit，计算与该 relic node 的相对位置
                for unit in friendly_units:
                    # 计算相对偏移
                    dx = int(unit['position'][0]) - int(relic_pos[0])
                    dy = int(unit['position'][1]) - int(relic_pos[1])
                    # 判断是否在规定范围内：假设有效区间为 dx,dy ∈ [–2, 2]
                    if -2 <= dx <= 2 and -2 <= dy <= 2:
                        # 将相对偏移映射到矩阵索引（例如索引 = (dx+2, dy+2)）
                        idx_x = dx + 2
                        idx_y = dy + 2
                        # 如果本步增速比上一步大，则标记该位置为 1
                        if new_delta > self.delta:
                            self.potential_point[relic_idx][idx_x][idx_y] = 1
        # 更新增速和 team_points，用于下一步比较
        self.delta = new_delta
        self.prev_team_points = new_team_points
    
        
        # 初始化 action_info 字典
        action_info_all = {i: {"sap_success": False, "sap_fail": False, "illegal_move": False} for i in range(self.max_units)}

        # 生成 action_info
        for unit_id in range(self.max_units):
            action = actions[unit_id]  # 当前 agent 的动作
            # 判断是否为 sap 动作
            if action[0] == 5:  # sap 动作
                if abs(action[1]) < self.unit_sap_range and abs(action[2]) < self.unit_sap_range:
                    # 检查该单位是否命中了敌方单位
                    target_pos = [action[1], action[2]]
                    hit_enemy = False
                    # 检查周围8个格子，目标范围内是否有敌方单位
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            target = [target_pos[0] + dx, target_pos[1] + dy]
                            if self.is_enemy_unit_at_position(unit_id, target, team_obs):  # 需要实现 is_enemy_unit_at_position 方法
                                hit_enemy = True
                    action_info_all[unit_id]["sap_success"] = hit_enemy
                    if not hit_enemy:
                        action_info_all[unit_id]["sap_fail"] = True
            # 判断非法移动
            if abs(action[1]) > self.unit_sap_range or abs(action[2]) > self.unit_sap_range:
                action_info_all[unit_id]["illegal_move"] = True
            if self.is_invalid_move(action, team_obs):  # 需要实现 is_invalid_move 方法
                action_info_all[unit_id]["illegal_move"] = True
        
        # 计算各 agent 的奖励
        rewards = []
        for unit_id in range(self.max_units):
            unit_reward = self.get_unit_reward(unit_id, team_obs, obs_agents, s_obs, action_info=action_info_all.get(unit_id, {}))
            rewards.append(unit_reward)

        self.prev_unit_energy = [obs_agents[i]["self_state"]["energy"] for i in range(self.max_units)]
        self.prev_sensor_count = np.sum(team_obs['sensor_mask'])
        self.prev_team_wins = team_obs['team_wins']
        
        # -------------------------------
        done = terminated or truncated
        dones = [done] * self.n_agents
        available_actions = self.action_space.copy()
        s_obs = [copy.deepcopy(team_obs) for _ in range(self.max_units)]
        self.current_step += 1
        return obs_agents, s_obs, rewards, dones, available_actions, enemy_obs_return ,info

    def is_enemy_unit_at_position(self, unit_id, target_pos, team_obs):
        """判断目标位置是否有敌方单位"""
        for i, mask in enumerate(team_obs['units_mask'][self.enemy_idx]):
            if mask:
                pos = team_obs['units']['position'][self.enemy_idx][i]
                if pos == target_pos:
                    return True
            return False

    def is_invalid_move(self, action, team_obs):
        """判断是否为非法移动"""
        # 假设某些条件定义了不可通行区域，如小行星或超出地图范围
        if abs(action[1]) >= self.unit_sap_range or abs(action[2]) >= self.unit_sap_range:
            return True
        target_tile = team_obs['map_features']['tile_type'][action[1], action[2]]
        if target_tile == 2:  # 假设 2 代表 Asteroid
            return True
        return False

    def seed(self, seed):
        """
        设置环境随机种子。
        """
        self.env.seed(seed)

    def render(self):
        """
        渲染环境（调用原环境的 render）。
        """
        self.env.render()

    def close(self):
        """
        关闭环境。
        """
        self.env.close()
