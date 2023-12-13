import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import robosuite as suite
import rlds


import logging


# ds = tfds.load('rlds/robosuite', split='train')

# Construct a tf.data.Dataset
print("----------dataset loading----------")
ds = tfds.load('mnist', split='train')
rl_ds = rlds.load('robosuite')
# ds = tfds.load('mnist', split='train', as_supervised=True, shuffle_files=True)
print("----------building pipeline----------")
# Build your input pipeline
ds = ds.shuffle(1000).batch(128).prefetch(10).take(5)
for image, label in ds:
  pass


print("----------creating robosuite env----------")
# init
env = suite.make(
    env_name="Lift", # Including other tasks, such as："Stack" and "Door"
    robots="Panda",  # Try other models, such as: "Sawyer" and "Jaco"
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
    # print("cycle:",i,"obs:", obs, "reward:", reward, "done:", done, "info:", info)
    logging.info(obs)
    logging.info(info)


print("----------exit----------")