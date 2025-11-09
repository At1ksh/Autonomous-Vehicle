import os, glob, json, csv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        base = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])  # Remove last FC layer
        self.fc = nn.Linear(512,128)# we compress to a 128 dim feature vector
        self.device = device
        self.to(device)
        self.eval() # freeze weights
        
    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        feat = self.fc(feat)
        return feat
        
        
# Camera Environment 

class CameraEnvResNet(gym.Env):
    def __init__(self, img_dir, events_dir, device="cpu", max_episode_len = 200):
        super(CameraEnvResNet, self).__init__()
        self.device = device
        self.img_dir = img_dir 
        self.events_dir = events_dir 
        self.ptr = 0
        self.episode_len = 0
        self.max_episode_len = max_episode_len 
        
        #Image List 
        self.image_files = sorted(glob.glob(os.path.join(img_dir,"*.jpg")))
        

        # Reward event files
        self.collision_file = os.path.join(events_dir, "collisions.csv")
        self.lane_file = os.path.join(events_dir, "lane_invasions.csv")

        # Track IDs of collision/lane invasion frames
        self.collision_frames = self._load_event_frames(self.collision_file)
        self.lane_frames = self._load_event_frames(self.lane_file)

        # Gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)  # forward, left, right, brake, idle

        # Initialize ResNet
        self.encoder = ResNetEncoder(device=device)
        
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])      
        
    
    def _load_event_frames(self, csv_path):
        if not os.path.exists(csv_path):
            return set()
        frames = set()
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = int(row.get("frame", -1))
                if frame >= 0:
                    frames.add(frame)
        return frames 


    # ------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ptr = np.random.randint(0, len(self.image_files))
        self.episode_len = 0
        obs = self._extract_features(self.image_files[self.ptr])
        return obs, {}

    def step(self, action):
        self.ptr += 1
        self.episode_len += 1

        # wrap index
        img_path = self.image_files[self.ptr % len(self.image_files)]
        obs = self._extract_features(img_path)

        frame_id = self.ptr % len(self.image_files)
        reward = self._compute_reward(frame_id, action)

        terminated = frame_id in self.collision_frames
        truncated = self.episode_len >= self.max_episode_len
        info = {}

        return obs, reward, terminated, truncated, info
    
    # ------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------
    def _extract_features(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0)
        feat = self.encoder(img_t).cpu().numpy().flatten().astype(np.float32)
        return feat

    # ------------------------------------------------------------
    # Reward Function
    # ------------------------------------------------------------
    def _compute_reward(self, frame_id, action):
        reward = 1.0
        if frame_id in self.collision_frames:
            reward -= 5.0
        if frame_id in self.lane_frames:
            reward -= 2.0
        reward -= 0.05 * abs(action - 2)  # small penalty for sharp steering
        return reward