import gymnasium as gym, cv2
import tensorflow as tf

env = gym.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    done = False
    state, info = env.reset()
    while not done:
        frame = env.render()
        cv2.imshow("CartPole", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        state , reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
env.close()