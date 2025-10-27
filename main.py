from mobile_robot import MobileRobot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

robot = MobileRobot(speed_max=10, accel_max=1, wheel_base=2.5)

x0 = np.array([0.0, 0.0, 0.0])

control = np.array([0.5, np.deg2rad(20)]) 

trajectory = robot.run(T=10, x0=x0, control=control)

print("Trajectory shape:", trajectory.shape)
print("First 5 states:\n", trajectory[:5])

x = trajectory[:, 0]
y = trajectory[:, 1]

plt.figure(figsize=(6,6))
plt.plot(x, y, label="Robot path")
plt.scatter(x[0], y[0], color="green", label="Start") 
plt.scatter(x[-1], y[-1], color="red", label="End")    
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Robot position over time")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.savefig("trajectory_positions.png")
plt.close()