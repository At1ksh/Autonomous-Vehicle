import math

class IDM:
    def __init__(self, v0=12.0, a=1.0,b=1.5,s0=2.0,T=1.2):
        self.v0, self.a, self.b, self.s0, self.T = v0, a, b, s0, T
    
    def accel(self, v, s, dv):
        # v: ego speed, s: gap to lead, dv: relative speed (ego - lead)
        s_star = self.s0 + max(0.0, v*self.T + (v*dv)/(2*math.sqrt(self.a*self.b)))
        return self.a * (1 - (v/self.v0)**4 - (s_star/max(s, 0.1))**2)

class PurePursuit:
    def __init__(self, L=2.8, lookahead=8.0):
        self.L, self.lookahead = L, lookahead
        
    def steering(self, ego_x, ego_y, ego_yaw, path_xy):
        target = None
        best = 1e9
        for px, py in path_xy:
            d = abs(px - self.lookahead)
            if d < best:
                best = d
                target = (px,py)
        if target is None:
            return 0
        
        x,y = target
        
        Ld2 = x*x + y*y
        if Ld2 < 1e-4:
            return 0
        kappa =2*y / Ld2
        
        return math.atan(self.L * kappa)