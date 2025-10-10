import numpy as np

class Kalman2D:
    def __init__(self):
        self.x = np.zeros((4,1), dtype=np.float32)   
        self.P = np.eye(4, dtype=np.float32) * 10
        dt = 0.1
        self.F = np.array([[1,0,dt,0],
                           [0,1,0,dt],
                           [0,0,1,0],
                           [0,0,0,1]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.1
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 0.5

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, zxy):
        z = np.array(zxy, dtype=np.float32).reshape(2,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
