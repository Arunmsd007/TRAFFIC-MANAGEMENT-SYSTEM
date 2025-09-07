# src/traffic.py
import random

class TrafficController:
    def __init__(self, use_rl=False):
        """
        Traffic Controller to manage signal phases
        :param use_rl: bool, whether to use Reinforcement Learning (RL) or Rule-based
        """
        self.use_rl = use_rl
        self.lane_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.last_green = "A"

        if self.use_rl:
            print("✅ RL Controller Enabled")
        else:
            print("⚡ Using Rule-based Controller")

    def update_lane(self, lane, count):
        """Update the count of vehicles in a lane"""
        self.lane_counts[lane] = count

    def decide_next(self, emergency_detected=False):
        """
        Decide which lane should get green signal
        :param emergency_detected: bool, whether an emergency vehicle is detected
        :return: (green_lane, phase)
        """
        if emergency_detected:
            # Emergency -> immediately clear the busiest lane
            green_lane = max(self.lane_counts, key=self.lane_counts.get)
            return green_lane, "green"

        if self.use_rl:
            # RL-based logic (simple placeholder, can connect to your DQN model)
            green_lane = self._rl_decision()
        else:
            # Rule-based: pick lane with max vehicles
            green_lane = max(self.lane_counts, key=self.lane_counts.get)

        self.last_green = green_lane
        return green_lane, "green"

    def _rl_decision(self):
        """
        Placeholder RL logic:
        Here you would integrate your DQN or stable-baselines agent.
        For now, it randomly chooses a lane (for testing).
        """
        lanes = list(self.lane_counts.keys())
        return random.choice(lanes)
