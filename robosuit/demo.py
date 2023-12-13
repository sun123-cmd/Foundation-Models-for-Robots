import numpy as np
import robosuite as suite
import logging
import tensorflow as tf
# 配置日志
logging.basicConfig(filename='example.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
"""
# 示例日志记录
logging.debug('This is a debug message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
"""

# init
env = suite.make(
    env_name="Lift", # Including other tasks："Stack" and "Door"
    robots="Panda",  # 尝试其他机器人模型，比如："Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

# runtime of robot module
for i in range(10):
    action = np.random.randn(env.robots[0].dof) # randokm actions
    obs, reward, done, info = env.step(action)  # get values
    env.render()  # render on display
    print("cycle:",i,"obs:", obs, "reward:", reward, "done:", done, "info:", info)
    logging.info(obs)
    logging.info(info)

"""
step() function takes an action as input and returns a tuple of (obs, reward, done, info) ,
obs is an OrderedDict containing observations [(name_string, np.array), ...], 
reward is the immediate reward obtained per step, 
done is a Boolean flag indicating if the episode has terminated,
info is a dictionary which contains additional metadata.
"""

