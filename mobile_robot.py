
from roboticstoolbox import VehicleBase
import numpy as np
class MobileRobot(VehicleBase):
    def __init__(self, speed_max=10, accel_max=1, wheel_base=3, track_width=1, speed=0):
        super().__init__(speed_max=speed_max, accel_max=accel_max)
        self.wheel_base = wheel_base
        self.track_width = track_width
        self.speed = speed
        self.min_steering = -np.deg2rad(30)
        self.max_steering = np.deg2rad(30)
        self.u_limited = self.u_limited
    
    def u_limited(self, u):
        accel, steering = u
        accel = np.clip(accel, -self.accel_max, self.accel_max)
        steering = np.clip(steering, self.min_steering, self.max_steering)
        return np.array([accel, steering])
    
    def deriv(self, state, control):
        x, y, heading = state
        accel, steering = control

        self.speed += accel * 0.1

        dx = self.speed * np.cos(heading)
        dy = self.speed * np.sin(heading)
        dheading = self.speed / self.wheel_base * np.tan(steering)

        return np.array([dx, dy, dheading])

