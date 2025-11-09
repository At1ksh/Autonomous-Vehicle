import os
import glob
import cv2
import torch
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from torch import nn


# ---------------------------------------------------------------
# Simple CNN Encoder for Camera Features
# ---------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------
# Camera RL Environment
# ---------------------------------------------------------------
class CameraEnv(gym.Env):
    def __init__(self, image_dir, events_dir, max_episode_len=200, device="cpu"):
        super(CameraEnv, self).__init__()
        self.image_dir = image_dir
        self.events_dir = events_dir
        self.device = device
        self.ptr = 0
        self.episode_len = 0
        self.max_episode_len = max_episode_len

        # Load all camera images
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if not self.image_files:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

        # Load events CSVs
        self.collisions = self._load_csv("collisions.csv")
        self.lane_invasions = self._load_csv("lane_invasions.csv")

        # Feature extractor (CNN)
        self.encoder = SimpleCNN().to(device)
        self.encoder.eval()

        # Observation: 128D CNN embedding
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
        # Actions: [0: forward, 1: left, 2: right, 3: brake, 4: idle]
        self.action_space = spaces.Discrete(5)

    # ---------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = np.random.randint(0, len(self.image_files))
        self.episode_len = 0
        obs = self._get_features(self.image_files[self.ptr])
        return obs, {}

    def step(self, action):
        self.ptr = (self.ptr + 1) % len(self.image_files)
        self.episode_len += 1

        obs = self._get_features(self.image_files[self.ptr])
        reward = float(self._compute_reward(self.image_files[self.ptr]))

        terminated = self.ptr == len(self.image_files) - 1
        truncated = self.episode_len >= self.max_episode_len
        info = {}

        return obs, reward, terminated, truncated, info

    # ---------------------------------------------------------------
    # Reward Computation
    # ---------------------------------------------------------------
    def _compute_reward(self, image_path):
        frame_id = self._extract_frame_id(image_path)
        reward = 1.0  # baseline reward for moving forward

        if frame_id in self.collisions:
            reward -= 5.0
        if frame_id in self.lane_invasions:
            reward -= 2.0

        return reward

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _get_features(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.encoder(img).cpu().numpy().flatten().astype(np.float32)
        return feat

    def _extract_frame_id(self, path):
        base = os.path.basename(path)
        num_str = ''.join(filter(str.isdigit, base))
        return int(num_str) if num_str else -1

    def _load_csv(self, filename):
        path = os.path.join(self.events_dir, filename)
        if not os.path.exists(path):
            return set()
        df = pd.read_csv(path)
        if "frame" in df.columns:
            return set(df["frame"].astype(int).tolist())
        return set()
