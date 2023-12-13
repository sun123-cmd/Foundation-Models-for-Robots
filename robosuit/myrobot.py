import rlds
import tensorflow_datasets as tfds


"""
1. 安装rlds和tfds的库，以及robosuite的依赖库
2. 从tfds中加载robosuite数据集，例如robosuite_panda_pick_place_can1。您可以使用tfds.load函数来加载数据集，指定split和shuffle_files参数。您也可以使用as_supervised参数来获取观察和奖励的元组。
3. 使用rlds的函数来操作数据集，例如rlds.map_episode或rlds.map_step2。您可以使用这些函数来对每个步骤或每个片段应用自定义的转换，例如提取特征或增强数据。
4. 使用rlds的函数来访问数据集的元数据，例如rlds.get_episode_metadata或rlds.get_step_metadata2。您可以使用这些函数来获取每个步骤或每个片段的额外信息，例如是否完成了任务或是否是第一个或最后一个步骤
"""

# Load the robosuite dataset from tfds
dataset = tfds.load("robosuite_panda_pick_place_can", split="train", shuffle_files=True)

# Define a custom function to extract the image and the action from each step
def extract_image_and_action(step):
  return step["image"], step["action"]

# Apply the custom function to each step using rlds.map_step
dataset = rlds.map_step(dataset, extract_image_and_action)

# Iterate over the episodes and steps in the dataset
for episode in dataset:
  # Get the episode metadata using rlds.get_episode_metadata
  episode_metadata = rlds.get_episode_metadata(episode)
  print(f"Episode {episode_metadata['episode_index']}")

  for step in episode["steps"]:
    # Get the step metadata using rlds.get_step_metadata
    step_metadata = rlds.get_step_metadata(step)
    print(f"Step {step_metadata['step_index']}")

    # Get the image and the action from the step
    image, action = step["data"]
    print(f"Image shape: {image.shape}")
    print(f"Action: {action}")
