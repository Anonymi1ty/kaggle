import numpy as np
import gym
from luxai_s3.wrappers import LuxAIS3GymEnv

class HarlLuxAIS3Env:
    def __init__(self, seed=42, player_id="player_0"):
        # 实例化原环境（可根据需要选择 numpy_output 参数）
        self.env = LuxAIS3GymEnv(numpy_output=True)
        obs, info = self.env.reset(seed=seed)
        env_cfg = info["params"]
        
        # 保存 player_id 及友/敌索引
        self.player_id = player_id
        if player_id == 'player_0':
            self.friendly_idx, self.enemy_idx = 0, 1
        else:
            self.friendly_idx, self.enemy_idx = 1, 0

        # 存储游戏相关参数，便于后续 reset/step 中使用
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.max_relic_nodes = info["full_params"]["max_relic_nodes"]
        self.max_point_nodes = 6
        self.local_window_size = env_cfg["unit_sensor_range"] * 2 + 1
        
        # 假设每队最多单位数为 16
        self.max_units = 16  
        self.n_agents = self.max_units * 2  # 两队共 32 个 agent
        
        # 原环境动作空间为每队一个 (max_units, 3) 的 Box，
        # 这里拆分为每个 agent 一个 (3,) 的 Box（取值范围 0~5）
        self.action_space = [gym.spaces.Box(low=0, high=5, shape=(3,), dtype=np.int16)
                             for _ in range(self.n_agents)]
        
        # 构造共享观测空间（例如全局地图信息等），这里直接用初始观测中的数据作为占位
        friendly_obs = obs[self.friendly_idx]
        enemy_obs = obs[self.enemy_idx]
        self.share_observation_space = [friendly_obs] * self.max_units + [enemy_obs] * self.max_units
        
        # 构造每个 agent 的观测空间
        self.observation_space = []
        for unit_id in range(env_cfg["max_units"]):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(friendly_obs, player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            self.observation_space.append(unit_obs)
        for unit_id in range(env_cfg["max_units"]):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(enemy_obs, player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            self.observation_space.append(unit_obs)
    
    @staticmethod
    def construct_unit_obs(player_obs, player_id, unit_id,
                             unit_sap_range, unit_move_cost, unit_sap_cost,
                             max_relic_nodes, max_point_nodes,
                             local_window_size=5):
        """
        构造当前单位的局部观测信息
        
        参数：
          player_obs: 当前玩家的观测字典（如 obs[player_id]）
          player_id: 当前玩家的 id，例如 "player_0" 或 "player_1"
          unit_id: 当前需要构造观测的单位在友方列表中的索引
          其它参数为游戏相关参数
          local_window_size: 用于提取局部地图信息的窗口大小（默认 5x5）
        
        返回：
          包含 unit_obs 信息的字典
        """
        # 判断友方和敌方索引
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
        for i in range(N):
            if player_obs['units_mask'][enemy_idx][i]:
                enemy_units.append({
                    'position': player_obs['units']['position'][enemy_idx][i],
                    'energy': player_obs['units']['energy'][enemy_idx][i]
                })

        # 提取当前单位状态（假设 unit_id 是合法且存在的）
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

        # 初始化 potential_point 矩阵，形状为 (max_relic_nodes, 5, 5)
        potential_point = [[[None for _ in range(5)] for _ in range(5)]
                           for _ in range(max_relic_nodes)]

        # 敌方单位信息：简单判断是否在 sensor_mask 内可见
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
        重置环境，返回 agent 级别观测、全局状态以及可用动作信息。
        """
        obs, info = self.env.reset(seed=seed)
        friendly_obs = obs[self.friendly_idx]
        enemy_obs = obs[self.enemy_idx]
        
        obs_agents = []
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(friendly_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            obs_agents.append(unit_obs)
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(enemy_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            obs_agents.append(unit_obs)
        
        #TODO: 确认state??
        state = info.get("state", None)
        available_actions = self.action_space.copy()
        return obs_agents, state, available_actions

    #TODO: 设计step函数和奖励函数
    def step(self, actions):
        """
        执行一步环境交互。
        参数 actions 为长度为 n_agents 的动作列表，
        返回 agent 级别观测、全局状态、各 agent 奖励、done 标志、info 和可用动作。
        """
        # 将动作列表拆分为友方和敌方（各 self.max_units 个）
        actions_team_friendly = np.stack(actions[:self.max_units])
        actions_team_enemy = np.stack(actions[self.max_units:])
        
        # 根据 player_id 确定环境动作字典的映射
        if self.player_id == "player_0":
            action_dict = {"player_0": actions_team_friendly, "player_1": actions_team_enemy}
        else:
            action_dict = {"player_0": actions_team_enemy, "player_1": actions_team_friendly}
        
        obs, reward, terminated, truncated, info = self.env.step(action_dict)
        
        friendly_obs = obs[self.friendly_idx]
        enemy_obs = obs[self.enemy_idx]
        
        obs_agents = []
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(friendly_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            obs_agents.append(unit_obs)
        for unit_id in range(self.max_units):
            unit_obs = HarlLuxAIS3Env.construct_unit_obs(enemy_obs, self.player_id, unit_id,
                                                         self.unit_sap_range, self.unit_move_cost, self.unit_sap_cost,
                                                         self.max_relic_nodes, self.max_point_nodes,
                                                         self.local_window_size)
            obs_agents.append(unit_obs)
        
        # 处理奖励：假设 reward 为各队奖励，分配给各自的 agent
        if isinstance(reward, (list, np.ndarray)) and len(reward) >= 2:
            reward_friendly = reward[self.friendly_idx]
            reward_enemy = reward[self.enemy_idx]
        else:
            reward_friendly = reward
            reward_enemy = reward
        rewards = [reward_friendly] * self.max_units + [reward_enemy] * self.max_units
        
        done = terminated or truncated
        dones = [done] * self.n_agents
        
        available_actions = self.action_space.copy()
        state = info.get("state", None)
        return obs_agents, state, rewards, dones, info, available_actions

    def seed(self, seed):
        """
        设置环境随机种子。
        """
        self.env.seed(seed)

    def render(self):
        """
        渲染环境（调用原环境 render）。
        """
        self.env.render()

    def close(self):
        """
        关闭环境。
        """
        self.env.close()
