%%writefile agent/agent.py
import numpy as np
import os
from stable_baselines3 import PPO
from lux.utils import direction_to  # 假设该函数用于计算两个位置间的移动方向
from base import Global, warp_point  # 引入 Global 常量和 warp_point 函数

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.env_cfg = env_cfg
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # 保存遗迹相关信息等（原有逻辑）
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # 加载训练好的 PPO 模型
        self.ppo_model = PPO.load("/kaggle/working/ppo_game_env_model")

    def compute_team_vision(self, tile_map, agent_positions):
        """
        根据所有己方单位的位置计算团队视野：
          - 对于每个单位，在其传感器范围内（范围为 -sensor_range ~ sensor_range），
            对 (x+dx, y+dy) 的贡献为：sensor_range + 1 - max(|dx|, |dy|)。
          - 如果目标地块为星云（tile_map == 1），则贡献减去 nebula_reduction（这里取 2，可根据实际参数调整）。
          - 将所有单位的贡献累加后，累加值大于 0 的地块视为可见。
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2  # 可根据实际情况调整
        vision = np.zeros(tile_map.shape, dtype=np.float32)
        # agent_positions 是一个列表，每个元素为 (x, y)（均为整数）
        for (x, y) in agent_positions:
            for dy in range(-sensor_range, sensor_range + 1):
                for dx in range(-sensor_range, sensor_range + 1):
                    new_x, new_y = warp_point(x + dx, y + dy)
                    contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                    # 如果该地块为星云，减去视觉削减值
                    if tile_map[new_y, new_x] == 1:
                        contrib -= nebula_reduction
                    vision[new_y, new_x] += contrib
        visible_mask = vision > 0
        return visible_mask

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # 提取原始 Lux 观测数据
        unit_mask = np.array(obs["units_mask"][self.team_id])          # (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # 每个单位的 [x, y]
        unit_energys = np.array(obs["units"]["energy"][self.team_id])     # 单位能量信息
        observed_relic_node_positions = np.array(obs["relic_nodes"])        # (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])       # (max_relic_nodes,)
        
        # 从 obs 中获取局部地图信息：
        # 由于比赛不会返回完整 tile_map，我们用 sensor_mask 和 map_features 构造局部 tile_map
        map_height = self.env_cfg["map_height"]
        map_width = self.env_cfg["map_width"]
        sensor_mask = np.array(obs["sensor_mask"])  # 布尔矩阵，形状 (map_height, map_width)
        tile_type_obs = np.array(obs["map_features"]["tile_type"])  # 真实 tile 类型，形状 (map_height, map_width)
        # 对于可见区域，tile_map 取真实 tile_type；不可见区域填 -1 表示未知
        tile_map = np.where(sensor_mask, tile_type_obs, -1)
        
        # 初始化返回动作数组
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)
        available_unit_ids = np.where(unit_mask)[0]
        
        # 针对每个可控单位，构造与训练时一致的全局观察（形状：(map_height, map_width, 3)）
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]    # [x, y]
            unit_energy = unit_energys[unit_id]     # 单位能量（本示例未直接使用）
            
            # 构造全局观察 obs_grid，形状为 (map_height, map_width, 3)
            obs_grid = np.zeros((map_height, map_width, 3), dtype=np.float32)
            
            # --- 通道 0：tile_map ---
            # 使用上面构造的 tile_map（对于不可见区域填 -1）
            obs_grid[..., 0] = tile_map
            
            # --- 通道 1：relic_map ---
            # 初始化 relic_map（全 0），然后更新在视野内的 relic 节点
            relic_map = np.zeros((map_height, map_width), dtype=np.int8)
            for i in range(len(observed_relic_node_positions)):
                if observed_relic_nodes_mask[i]:
                    x, y = observed_relic_node_positions[i]
                    x, y = int(x), int(y)
                    if 0 <= x < map_width and 0 <= y < map_height:
                        # 仅在可见区域内更新遗迹信息
                        if sensor_mask[y, x]:
                            relic_map[y, x] = 1
            obs_grid[..., 1] = relic_map
            
            # --- 通道 2：agent_position ---
            # 仅标记当前单位的位置（如果该位置可见）
            agent_layer = np.zeros((map_height, map_width), dtype=np.int8)
            unit_x = int(unit_pos[0])
            unit_y = int(unit_pos[1])
            if 0 <= unit_x < map_width and 0 <= unit_y < map_height and sensor_mask[unit_y, unit_x]:
                agent_layer[unit_y, unit_x] = 1
            obs_grid[..., 2] = agent_layer
            
            # 扩展 batch 维度，形状变为 (1, map_height, map_width, 3)
            state = obs_grid[np.newaxis, ...]
            # 调用 PPO 模型进行预测，deterministic=True 保证输出确定性动作
            action, _ = self.ppo_model.predict(state, deterministic=True)
            action = int(action)
            # 将模型输出动作映射为比赛要求的格式（此处简单设为 [action, 0, 0]）
            actions[unit_id] = [action, 0, 0]
        
        return actions
