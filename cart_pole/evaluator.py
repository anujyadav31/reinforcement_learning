import gymnasium as gym, cv2
import tensorflow as tf
from keras.models import load_model

env = gym.make("CartPole-v1", render_mode="rgb_array")
q_net = load_model("q_learning_q_net.keras")


def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action


for episode in range(5):
    done = False
    state = tf.convert_to_tensor([env.reset()[0]])
    while not done:
        frame = env.render()
        cv2.imshow("CartPole", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(100)
        action = policy(state)
        state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        state = tf.convert_to_tensor([state])
env.close()
