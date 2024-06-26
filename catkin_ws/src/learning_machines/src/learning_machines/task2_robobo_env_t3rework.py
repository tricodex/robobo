import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps, CallbackList
from datetime import datetime
from robobo_interface import IRobobo, Position, Orientation
from data_files import FIGRURES_DIR

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo, ep_steps=1024, simulation=True):
        super(RoboboEnv, self).__init__()
        self.robot = rob
        self.simulation = simulation
        self.ep_steps = ep_steps
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        
        self.has_reset = False
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset parent class
        if self.simulation:
            self.robot.stop_simulation()
            self.robot.set_position(Position(0, 0, 0), Orientation(0, 0, 0))
            self.robot.play_simulation()
        
        self.robot.set_phone_tilt(240, 50)  # tilted down
        self.robot.sleep(3)
        self.current_step = 0
        self.has_reset = True
        return self._get_obs(), {}

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before stepping")
        
        self.current_step += 1
        
        action[2] = (((action[2] + 1) / 2) * 0.75) + 0.15
        left_speed, right_speed, duration_scale = action * np.array([50, 50, 800])
        
        self.robot.move(left_speed, right_speed, duration_scale)
        self.robot.sleep(action[2])
        
        obs = self._get_obs()
        reward = self._calculate_reward(obs)
        print(self.current_step,reward)
        
        done = self.current_step >= self.ep_steps
        
        return obs, reward, done, False, {}

    def _get_obs(self):
        frame = self.robot.get_image_front()
        frame = cv2.resize(frame, (64, 64))
        return frame
    

    def _calculate_reward(self, obs):
        food_collected = self.robot.nr_food_collected()
        
        # Convert to HSV for easier color detection
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
        
        # Detect green area (representing food)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_area = np.sum(green_mask > 0)
        
        # Calculate centroid of the green area
        green_centroid = np.mean(np.argwhere(green_mask > 0), axis=0) if green_area > 0 else np.array([32, 32])
        
        # Distance to the green centroid
        distance_to_green = np.linalg.norm(np.array([32, 32]) - green_centroid)
        
        # Reward for collecting food
        reward = food_collected * 10  # Reward for collecting food
        
        # Additional reward for getting closer to the green area
        if green_area > 0:
            reward += 5 / (distance_to_green + 1)  # Inverse of the distance to the green area for reward
        
        # Less reward if no food detected by base
        if not self.robot.base_detects_food():
            reward *= 0.5
        
        return reward

def run(rob: IRobobo):
    # load_model = False
    env = RoboboEnv(rob)
    env = DummyVecEnv([lambda: env])
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(FIGRURES_DIR, "models", f"PPO_{current_datetime}")
    os.makedirs(log_dir, exist_ok=True)
    
    steps = 1024
    episodes = 1000
    total_timesteps = steps * episodes
    
    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps // 1000, save_path=log_dir, name_prefix="ppo_model")
    event_callback = EveryNTimesteps(n_steps=total_timesteps // 1000, callback=checkpoint_callback)
    
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=1e-4, 
                n_steps=512, 
                batch_size=128, 
                n_epochs=5, 
                gamma=0.99, 
                ent_coef=0.01, 
                use_sde=True, 
                sde_sample_freq=4, 
                target_kl=0.015)
    
    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=steps * 11,
                                 deterministic=True, render=False)
    
    callbacks = CallbackList([checkpoint_callback, event_callback, eval_callback])
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(os.path.join(log_dir, "final_model"))

