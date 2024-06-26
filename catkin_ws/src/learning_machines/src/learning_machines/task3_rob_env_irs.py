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

stream = True

class RoboboEnv(gym.Env):
    def __init__(self, rob: IRobobo, ep_steps=1024, simulation=True):
        super(RoboboEnv, self).__init__()
        self.robot = rob
        self.simulation = simulation
        self.ep_steps = ep_steps
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Extend observation space to include IR sensor readings
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'ir_sensors': spaces.Box(low=0, high=1000, shape=(8,), dtype=np.float32)
        })
        
        self.has_reset = False
        self.current_step = 0
        self.consecutive_red_sightings = 0
        self.previous_action = np.zeros(3)
        
        # Curriculum stage
        self.curriculum_stage = 0
        self.stages = [
            "find_red_box",
            "push_red_box",
            "find_green_box",
            "push_to_green_box"
        ]
        if stream:
        
            # Create a named window for streaming
            cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.simulation:
            self.robot.stop_simulation()
            self.robot.set_position(Position(-2.4000000000000026, 0.07699995934963236, 0.03970504179596901), Orientation(-1.57191039448275, -1.5144899542893442, -1.5719101704888712))
            self.robot.play_simulation()
        
        self.robot.set_phone_tilt(240, 50)  # tilted down
        self.robot.sleep(3)
        self.current_step = 0
        self.consecutive_red_sightings = 0
        self.has_reset = True
        self.previous_action = np.zeros(3)
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
        if stream:
         self._stream_frame(obs['image'])
        reward = self._calculate_reward(obs, action)
        print(f"Step: {self.current_step}, Reward: {reward}, Stage: {self.stages[self.curriculum_stage]}")
        
        done = self.current_step >= self.ep_steps
        
        self.previous_action = action
        
        return obs, reward, done, False, {}

    def _get_obs(self):
        frame = self.robot.get_image_front()
        # Crop the frame to focus on the bottom part
        height, width = frame.shape[:2]
        crop_height = int(height * 0.6)  # Crop the top 40%
        frame = frame[height - crop_height:height, :]
        frame = cv2.resize(frame, (64, 64))
        
        # Get IR sensor readings
        ir_sensors = np.array(self.robot.read_irs(), dtype=np.float32)
        
        return {
            'image': frame,
            'ir_sensors': ir_sensors
        }
    
    def _stream_frame(self, frame):
        cv2.imshow("Robot View", cv2.resize(frame, (320, 320)))
        cv2.waitKey(1)
        
    def close(self):
        if stream:
            cv2.destroyAllWindows()

    def _calculate_reward(self, obs, action):
        frame = obs['image']
        ir_sensors = obs['ir_sensors']
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
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
        
        reward = 0
        
        if self.curriculum_stage == 0:  # find_red_box
            if red_area > 0:
                self.consecutive_red_sightings += 1
                reward += 10 * self.consecutive_red_sightings
            else:
                self.consecutive_red_sightings = 0
                reward -= 1
        
        elif self.curriculum_stage == 1:  # push_red_box
            if red_area > 0:
                reward += 5
                front_ir = max(ir_sensors[2], ir_sensors[3])  # Front IR sensors
                if front_ir > 100:
                    reward += 20  # High reward for being close to the red box
            else:
                reward -= 5
        
        elif self.curriculum_stage == 2:  # find_green_box
            if red_area > 0 and green_area > 0:
                reward += 20
            elif red_area > 0:
                reward += 5
            else:
                reward -= 5
        
        elif self.curriculum_stage == 3:  # push_to_green_box
            if red_area > 0 and green_area > 0:
                red_distance = np.linalg.norm(red_centroid - green_centroid)
                reward += 50 / (red_distance + 1)
            elif red_area > 0:
                reward += 5
            else:
                reward -= 10
        
        # Penalty for unnecessary movement
        movement_penalty = np.sum(np.abs(action - self.previous_action))
        reward -= movement_penalty * 2
        
        # Encourage forward movement
        if action[0] > 0 and action[1] > 0:
            reward += 1
        
        # Penalty for spinning
        if np.abs(action[0] - action[1]) > 0.5:
            reward -= 5
        
        return reward

    def increase_difficulty(self):
        if self.curriculum_stage < len(self.stages) - 1:
            self.curriculum_stage += 1
            print(f"Progressing to stage: {self.stages[self.curriculum_stage]}")
        else:
            print("Maximum stage reached")

    def get_current_stage(self):
        return self.stages[self.curriculum_stage]

def run(rob: IRobobo):
    load_model = False  # Set to True if you want to load a pre-trained model
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
    checkpoint_callback = CheckpointCallback(save_freq=total_timesteps // 100, save_path=log_dir, name_prefix="ppo_model")
    event_callback = EveryNTimesteps(n_steps=total_timesteps // 100, callback=checkpoint_callback)
    eval_callback = EvalCallback(env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=steps * 11,
                                 deterministic=True, render=False)
    callbacks = CallbackList([checkpoint_callback, event_callback, eval_callback])

    if not load_model:
        # Create new model with curriculum learning
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir,
                    learning_rate=lambda f: f * 1e-3,  # Learning rate annealing
                    n_steps=512,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    ent_coef=0.01,
                    use_sde=True,
                    sde_sample_freq=4,
                    target_kl=0.015,
                    clip_range=lambda f: f * 0.2)  # Clip range annealing
    else:
        # Load pre-trained model
        path = os.path.join(FIGRURES_DIR, "models", "PPO_2024-06-25_21-23-14", "ppo_model_10231_steps")
        model = PPO.load(path, env=env)

    # Curriculum learning
    for i in range(4):  # 4 stages of curriculum
        current_stage = env.env_method("get_current_stage")[0]
        print(f"Curriculum stage {i+1}/4: {current_stage}")
        model.learn(total_timesteps=total_timesteps // 4, callback=callbacks)
        # Progress to the next stage
        env.env_method("increase_difficulty")

    # Save the final model
    model.save(os.path.join(log_dir, "final_model"))
    
# Position:  -2.4000000000000026 0.07699995934963236 0.03970504179596901
# Orientation:  -1.57191039448275 -1.5144899542893442 -1.5719101704888712