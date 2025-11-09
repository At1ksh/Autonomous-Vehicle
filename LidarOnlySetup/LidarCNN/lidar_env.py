import os
import json
import glob
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LidarEnv(gym.Env):
    """
    LiDAR-only Reinforcement Learning Environment
    ------------------------------------------------
    Works in two modes:
      - 'offline': loads LiDAR .npy + label .json pairs
      - 'carla': expects live CARLA sensor callbacks (to be added later)
    """

    def __init__(self, lidar_dir, label_dir=None, mode='offline', max_episode_len=200):
        super(LidarEnv, self).__init__()
        self.mode = mode
        self.lidar_dir = lidar_dir
        self.label_dir = label_dir or lidar_dir
        self.ptr = 0
        self.episode_len = 0
        self.max_episode_len = max_episode_len

        # === Load files ===
        self.lidar_files = sorted(glob.glob(os.path.join(lidar_dir, "*.npy")))
        self.label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.json")))

        if len(self.label_files) != len(self.lidar_files):
            print(f"⚠️ Warning: Found {len(self.lidar_files)} LiDAR files and {len(self.label_files)} label files.")
            print("Matching will be done by filename prefix (e.g., lidar_top_000832.npy ↔ labels_000832.json)")

        # === RL spaces ===
        self.num_bins = 72
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.num_bins,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    # ---------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = np.random.randint(0, len(self.lidar_files))
        self.episode_len = 0
        self.current_obs = self._extract_features(self.lidar_files[self.ptr])
        return self.current_obs, {}


    def step(self, action):
        self.ptr += 1
        self.episode_len += 1

        # ✅ wrap index safely
        idx = self.ptr % len(self.lidar_files)

        obs = self._extract_features(self.lidar_files[idx])
        label_path = self._get_label_path(self.lidar_files[idx])

        reward = float(self._compute_reward(action, label_path))

        terminated = False
        truncated = self.episode_len >= self.max_episode_len

        # if you want to end after a full pass through the dataset
        if self.ptr >= len(self.lidar_files) - 1:
            terminated = True

        info = {}
        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------
    # Feature Extraction
    # ---------------------------------------------------------------

    def _extract_features(self, npy_path):
        """Convert LiDAR point cloud to normalized distance histogram"""
        lidar = np.load(npy_path)
        if lidar.ndim == 2 and lidar.shape[1] >= 3:
            dists = np.linalg.norm(lidar[:, :2], axis=1)
        else:
            dists = np.abs(lidar.flatten())

        bins, _ = np.histogram(dists, bins=self.num_bins, range=(0, 50))
        bins = bins.astype(np.float32)
        bins /= (bins.max() + 1e-6)
        return bins

    # ---------------------------------------------------------------
    # Reward Function
    # ---------------------------------------------------------------

    def _compute_reward(self, action, label_path):
        """
        Compute reward using label metadata.
        Expected JSON keys (if available):
            'collision': bool
            'lane_offset': float
            'speed': float
        """
        labels = json.load(open(label_path))

        collision = float(labels.get("collision", 0))
        lane_offset = float(labels.get("lane_offset", 0))
        speed = float(labels.get("speed", 0))

        # Basic reward shaping
        reward = 0.0
        reward += +1.0 * (1.0 - min(lane_offset, 1.0))   # stay centered
        reward += +0.1 * speed                           # encourage movement
        reward -= +5.0 * collision                       # heavy penalty
        reward -= +0.05 * abs(action - 2)                # mild penalty for turning

        # ✅ Return only the scalar reward (no tuple!)
        return reward


    # ---------------------------------------------------------------
    # Optional (future) CARLA mode hooks
    # ---------------------------------------------------------------

    def attach_carla_sensors(self, carla_world, ego_vehicle):
        """
        (Placeholder) To be filled when using CARLA live mode.
        Attach collision, lane, and LiDAR sensors and set callbacks.
        """
        raise NotImplementedError("CARLA live mode integration not yet implemented.")

    def _get_label_path(self, npy_path):
        base = os.path.basename(npy_path)
        label_name = base.replace("lidar_top_", "labels_").replace(".npy", ".json")
        label_path = os.path.join(self.label_dir, label_name)
        return label_path if os.path.exists(label_path) else None
