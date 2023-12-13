import numpy as np
import tensorflow as tf
import gym
from collections import deque

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 设置参数
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
episodes = 1000
memory = deque(maxlen=2000)

# 构建深度 Q 网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# 训练 DQN 网络
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, -1])  # 修改这一行
    total_reward = 0

    for time in range(500):  # 最多运行500步
        # 选择动作
        if np.random.rand() <= 0.1:  # 使用epsilon-greedy策略
            action = env.action_space.sample()  # 随机选择动作
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        # 执行动作
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, -1])  # 修改这一行

        # 存储经验到记忆
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # 从记忆中随机抽取样本进行训练
        if len(memory) > batch_size:
            minibatch = np.array(random.sample(memory, batch_size))
            states = np.vstack(minibatch[:, 0])
            actions = minibatch[:, 1]
            rewards = minibatch[:, 2]
            next_states = np.vstack(minibatch[:, 3])
            dones = minibatch[:, 4]

            targets = rewards + 0.95 * np.max(model.predict(next_states), axis=1) * (1 - dones)
            target_f = model.predict(states)
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]

            model.fit(states, target_f, epochs=1, verbose=0)

        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# 测试 DQN 网络
total_rewards = []
for _ in range(10):  # 运行10个测试回合
    state = env.reset()
    state = np.reshape(state, [1, -1])  # 修改这一行
    total_reward = 0

    for _ in range(500):  # 最多运行500步
        action = np.argmax(model.predict(state)[0])
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, -1])  # 修改这一行
        state = next_state
        total_reward += reward

        if done:
            break

    total_rewards.append(total_reward)

average_reward = np.mean(total_rewards)
print(f"Average Test Reward: {average_reward}")

# 关闭环境
env.close()
