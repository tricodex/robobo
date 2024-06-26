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
        self.consecutive_red_sightings = 0
        
        # Create a named window for streaming
        cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Reset parent class
        if self.simulation:
            self.robot.stop_simulation()
            self.robot.set_position(Position(-2.4000000000000026, 0.07699995934963236, 0.03970504179596901), Orientation(-1.57191039448275, -1.5144899542893442, -1.5719101704888712))
            self.robot.play_simulation()
        
        # Set phone tilt to a slightly downward angle
        self.robot.set_phone_tilt(240, 50)  # tilted down
        self.robot.sleep(3)
        self.current_step = 0
        self.consecutive_red_sightings = 0
        self.has_reset = True
        return self._get_obs(), {}

    def step(self, action):
        if not self.has_reset:
            raise RuntimeError("Environment must be reset before stepping")
        
        self.current_step += 1
        
        action[2] = (((action[2]+1)/2)*0.75)+0.15
        
        # Convert normalized actions to robot commands
        left_speed, right_speed, duration_scale = action * np.array([50, 50, 800])
        
        self.robot.move(left_speed, right_speed, duration_scale)
        self.robot.sleep(action[2])
        
        obs = self._get_obs()
        self._stream_frame(obs)
        reward = self._calculate_reward(obs)
        print(self.current_step, reward)
        
        done = self.current_step >= self.ep_steps
        
        return obs, reward, done, False, {}

    def _get_obs(self):
        frame = self.robot.get_image_front()
        # Crop the frame to focus on the bottom part
        height, width = frame.shape[:2]
        crop_height = int(height * 0.6)  # Crop the top 40%
        frame = frame[height - crop_height:height, :]
        frame = cv2.resize(frame, (64, 64))
        return frame
    
    def _stream_frame(self, frame):
        cv2.imshow("Robot View", cv2.resize(frame, (320, 320)))
        cv2.waitKey(1)
        
    def close(self):
        cv2.destroyAllWindows()

    def _calculate_reward(self, obs):
        hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
        irs = self.robot.read_irs()
        irs_value = max(irs[2], irs[3])  # Use IR sensors 2 and 3
        
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        red_area = np.sum(red_mask > 0)
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_area = np.sum(green_mask > 0)
        
        red_centroid = np.mean(np.argwhere(red_mask > 0), axis=0) if red_area > 0 else np.array([32, 32])
        green_centroid = np.mean(np.argwhere(green_mask > 0), axis=0) if green_area > 0 else np.array([32, 32])
        
        distance = np.linalg.norm(red_centroid - green_centroid)
        
        reward = -distance / 64.0
        
        red_distance = np.abs(red_centroid[0] - 32)
        reward += 1 / (red_distance + 1)
        
        if red_area > 0:
            green_distance = np.abs(green_centroid[0] - 32)
            reward += 1 / (green_distance + 1)
        
        # Scaled reward for seeing the red object
        if red_area > 0:
            self.consecutive_red_sightings += 1
            reward += 10 * self.consecutive_red_sightings
            if irs_value > 100:
                reward += 100 * self.consecutive_red_sightings
        else:
            self.consecutive_red_sightings = 0
        
        if green_area > 0:
            reward += 5
        
        if np.sum(red_mask & green_mask) > 0:
            reward += 5 * self.consecutive_red_sightings
            
        if red_area == 0 or green_area == 0:
            reward -= 10
        
        return reward


def run(rob: IRobobo):
    load_model = True
    steps = 1024
    episodes = 1000
    total_timesteps = steps * episodes

    # Common setup
    env = RoboboEnv(rob)
    env = DummyVecEnv([lambda: env])
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(FIGRURES_DIR, "models", f"PPO_{current_datetime}")
    os.makedirs(log_dir, exist_ok=True)

    # Callbacks setup
    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps // 1000, save_path=log_dir, name_prefix="ppo_model")
    event_callback = EveryNTimesteps(n_steps=total_timesteps // 1000, callback=checkpoint_callback)
    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=steps * 11,
                                 deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_callback, event_callback, eval_callback])

    if not load_model:
        # Create new model
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir,
                learning_rate=1e-4,  # Reduced learning rate for stability
                n_steps=512,  # Increased steps per update for better learning
                batch_size=128,  # Increased batch size
                n_epochs=5,  # Reduced epochs to prevent overfitting
                gamma=0.99,  # Keep default discount factor
                ent_coef=0.01,  # Slightly increased entropy coefficient for exploration
                use_sde=True,  # Use generalized State Dependent Exploration
                sde_sample_freq=4,  # Sample new noise matrix every 4 steps
                target_kl=0.015,)  # Set a target KL divergence to prevent too large updates
    else:
        # Load pre-trained model
        path = os.path.join(FIGRURES_DIR, "models", "PPO_2024-06-25_21-23-14", "ppo_model_10231_steps")
        model = PPO.load(path, env=env)

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))