import gym
import time

def get_user_action():
    action = input("Enter action (0 for left, 1 for right): ")
    try:
        action = int(action)
        if action not in [0, 1]:
            raise ValueError
    except ValueError:
        print("Invalid action. Please enter 0 or 1.")
        return get_user_action()
    return action

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset()
    env.render()
    done = False
    total_reward = 0

    while not done:
        action = get_user_action()
        observation, reward, done, truncated, info = env.step(action)
        done = done or truncated
        total_reward += reward
        env.render()
        time.sleep(0.1)  # slow down the rendering for better user experience

    print(f"Game Over! Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
