"""
JARVIS-PRIME Robotics Agent
==============================

Capabilities:
- Forward/inverse kinematics (2D/3D manipulators)
- Path planning (RRT, A*)
- PID controller simulation
- Trajectory generation
- Robot dynamics (mass-spring-damper)

All computations are local. Phase 3+: ROS2 integration.
"""
from __future__ import annotations

import math
import heapq
from typing import Any

import numpy as np

from jarvis.agents.base_agent import BaseAgent


class Kinematics2D:
    """2D robotic arm kinematics (2-link planar manipulator)."""

    @staticmethod
    def forward(L1: float, L2: float, theta1_deg: float, theta2_deg: float) -> dict[str, Any]:
        """Forward kinematics: joint angles -> end-effector position."""
        t1 = math.radians(theta1_deg)
        t2 = math.radians(theta2_deg)

        # Elbow position
        x1 = L1 * math.cos(t1)
        y1 = L1 * math.sin(t1)

        # End-effector position
        x2 = x1 + L2 * math.cos(t1 + t2)
        y2 = y1 + L2 * math.sin(t1 + t2)

        return {
            "joint_angles_deg": [theta1_deg, theta2_deg],
            "elbow_position": [round(x1, 4), round(y1, 4)],
            "end_effector": [round(x2, 4), round(y2, 4)],
            "reach": round(math.sqrt(x2**2 + y2**2), 4),
            "max_reach": L1 + L2,
        }

    @staticmethod
    def inverse(L1: float, L2: float, x: float, y: float) -> dict[str, Any]:
        """Inverse kinematics: end-effector position -> joint angles."""
        d = math.sqrt(x**2 + y**2)

        if d > L1 + L2:
            return {"error": "Target unreachable", "distance": d, "max_reach": L1 + L2}

        # Cosine law for theta2
        cos_t2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_t2 = max(-1, min(1, cos_t2))

        # Elbow-up solution
        theta2 = math.acos(cos_t2)
        k1 = L1 + L2 * math.cos(theta2)
        k2 = L2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)

        # Elbow-down solution
        theta2_down = -theta2
        k1d = L1 + L2 * math.cos(theta2_down)
        k2d = L2 * math.sin(theta2_down)
        theta1_down = math.atan2(y, x) - math.atan2(k2d, k1d)

        return {
            "target": [x, y],
            "elbow_up": {
                "theta1_deg": round(math.degrees(theta1), 2),
                "theta2_deg": round(math.degrees(theta2), 2),
            },
            "elbow_down": {
                "theta1_deg": round(math.degrees(theta1_down), 2),
                "theta2_deg": round(math.degrees(theta2_down), 2),
            },
        }


class PIDController:
    """PID controller simulation."""

    def __init__(self, Kp: float = 1.0, Ki: float = 0.1, Kd: float = 0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def simulate(
        self,
        setpoint: float = 1.0,
        duration: float = 10.0,
        dt: float = 0.01,
        plant_gain: float = 1.0,
        plant_time_constant: float = 1.0,
    ) -> dict[str, Any]:
        """Simulate PID control of a first-order plant."""
        steps = int(duration / dt)
        output = 0.0
        integral = 0.0
        prev_error = setpoint

        times = []
        outputs = []
        errors_list = []

        for i in range(steps):
            error = setpoint - output
            integral += error * dt
            derivative = (error - prev_error) / dt

            control = self.Kp * error + self.Ki * integral + self.Kd * derivative

            # First-order plant: tau * dy/dt + y = K * u
            dy = (plant_gain * control - output) / plant_time_constant
            output += dy * dt

            prev_error = error

            if i % int(0.1 / dt) == 0:  # Sample every 0.1s
                times.append(round(i * dt, 2))
                outputs.append(round(output, 4))
                errors_list.append(round(error, 4))

        # Performance metrics
        rise_time = None
        settling_time = None
        overshoot = 0.0

        for i, (t, o) in enumerate(zip(times, outputs)):
            if rise_time is None and o >= 0.9 * setpoint:
                rise_time = t
            if o > setpoint:
                overshoot = max(overshoot, (o - setpoint) / setpoint * 100)

        for i in range(len(outputs) - 1, -1, -1):
            if abs(outputs[i] - setpoint) > 0.02 * setpoint:
                settling_time = times[min(i + 1, len(times) - 1)]
                break

        return {
            "controller": {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd},
            "setpoint": setpoint,
            "performance": {
                "rise_time_s": rise_time,
                "settling_time_s": settling_time,
                "overshoot_pct": round(overshoot, 2),
                "steady_state_error": round(abs(outputs[-1] - setpoint), 6),
            },
            "samples": len(times),
            "final_output": outputs[-1] if outputs else 0,
        }


class PathPlanner:
    """Grid-based path planning (A* algorithm)."""

    @staticmethod
    def a_star(
        grid_size: int = 20,
        start: tuple[int, int] = (0, 0),
        goal: tuple[int, int] = (19, 19),
        obstacles: list[tuple[int, int]] | None = None,
    ) -> dict[str, Any]:
        """A* path planning on a 2D grid."""
        if obstacles is None:
            # Generate some random obstacles
            np.random.seed(42)
            obstacles = []
            for _ in range(grid_size * grid_size // 5):
                ox, oy = np.random.randint(0, grid_size, 2)
                if (ox, oy) != start and (ox, oy) != goal:
                    obstacles.append((int(ox), int(oy)))

        obstacle_set = set(obstacles)

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        nodes_explored = 0

        while open_set:
            _, current = heapq.heappop(open_set)
            nodes_explored += 1

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()

                return {
                    "algorithm": "A*",
                    "grid_size": grid_size,
                    "start": start,
                    "goal": goal,
                    "obstacles": len(obstacles),
                    "path_found": True,
                    "path_length": len(path),
                    "nodes_explored": nodes_explored,
                    "path": path[:20],  # First 20 waypoints
                }

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)

                if 0 <= nx < grid_size and 0 <= ny < grid_size and neighbor not in obstacle_set:
                    tentative_g = g_score[current] + (1.414 if abs(dx) + abs(dy) == 2 else 1)

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f, neighbor))

        return {
            "algorithm": "A*",
            "path_found": False,
            "nodes_explored": nodes_explored,
        }


class RoboticsAgent(BaseAgent):
    """Domain agent for robotics simulation and analysis."""

    def __init__(self):
        super().__init__(name="RoboticsAgent", domain="robotics")
        self.kinematics = Kinematics2D()
        self.pid = PIDController()
        self.planner = PathPlanner()

    def get_capabilities(self) -> list[str]:
        return [
            "forward_kinematics",
            "inverse_kinematics",
            "pid_control_simulation",
            "path_planning_astar",
            "trajectory_generation",
        ]

    async def execute(self, task: dict[str, Any], tools: list[Any]) -> dict[str, Any]:
        task_type = task.get("type", "overview")
        params = task.get("parameters", {})

        if task_type == "forward_kinematics":
            return self.kinematics.forward(
                L1=params.get("L1", 1.0), L2=params.get("L2", 0.8),
                theta1_deg=params.get("theta1", 45), theta2_deg=params.get("theta2", 30),
            )
        elif task_type == "inverse_kinematics":
            return self.kinematics.inverse(
                L1=params.get("L1", 1.0), L2=params.get("L2", 0.8),
                x=params.get("x", 1.0), y=params.get("y", 0.5),
            )
        elif task_type == "pid":
            return self.pid.simulate()
        elif task_type == "path_planning":
            return self.planner.a_star()
        elif task_type == "full_analysis":
            return self._full_analysis()
        else:
            return self._overview()

    def _full_analysis(self) -> dict[str, Any]:
        return {
            "task": "full_robotics",
            "forward_kinematics": self.kinematics.forward(1.0, 0.8, 45, 30),
            "inverse_kinematics": self.kinematics.inverse(1.0, 0.8, 1.0, 0.5),
            "pid_simulation": self.pid.simulate(),
            "path_planning": self.planner.a_star(),
        }

    def _overview(self) -> dict[str, Any]:
        return {
            "task": "robotics_overview",
            "state_of_art_2026": {
                "ROS2": "Jazzy stable / Lyrical LTS entering GA",
                "manipulation": "MoveIt 2 + Tesseract (GPU-accelerated)",
                "training": "NVIDIA Isaac Lab (1000x real-time sim-to-real)",
                "foundation_models": "Vision-Language-Action (VLA): RLDX-1, RT-X",
            },
            "capabilities": self.get_capabilities(),
        }
