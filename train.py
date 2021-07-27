import gym
from DDPG import DDPG
from utils import *
import time
import datetime


def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v3"
    log_interval = 10  # print avg reward after interval
    gamma = 0.99  # discount for future rewards
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 0.0005
    exploration_noise = 0.1
    tau = 0.005
    policy_noise = 0.2  # target policy smoothing noise
    noise_clip = 0.5
    max_episodes = 1000  # max num of episodes
    max_timesteps = 2000  # max timesteps in one episode
    directory = "./model"
    filename = "{}_{}".format(env_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    render_every = 100
    ###################################

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer()
    #######################################################################
    subBuffer = []
    n_save = 30
    #######################################################################

    # logging variables:
    rewards = []
    best_ep = -200
    best_reward = 0
    log_f = open("log.txt", "w+")
    start_time = time.time()

    # training procedure:
    #######################################################################
    # collect data for replay buffer
    print("Collecting initial data", end='')
    for episode in range(1, 101):
        if episode % 10 == 0:
            print(".", end='')
        state = env.reset()
        for t in range(200):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state

            if done:
                break
    print()
    for t in range(100):
        agent.learn(replay_buffer, batch_size, gamma, tau, policy_noise, noise_clip)
    #######################################################################
    for episode in range(1, max_episodes+1):

        state = env.reset()
        ep_reward = 0

        for t in range(max_timesteps):

            if episode % render_every == 0:
                env.render()

            # select action and add exploration noise:
            action = agent.choose_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)

            # take action in env:
            next_state, reward, done, _ = env.step(action)

            # replay_buffer.add((state, action, reward, next_state, float(done)))
            #######################################################################
            subBuffer.append([state, action, reward, next_state, float(done)])
            #######################################################################

            state = next_state
            ep_reward += reward

            #######################################################################
            if (t+1) % n_save == 0 or done:
                # if fall over
                if reward <= -10:
                    punishment = 0
                    max_punishment = 5
                    punishment_incre = max_punishment / n_save
                    for data in subBuffer:
                        punishment += punishment_incre
                        data[2] -= punishment

                # extract from subBuffer and add to replay buffer
                for _ in range(len(subBuffer)):
                    temp = tuple(subBuffer.pop())
                    replay_buffer.add(temp)
            #######################################################################

            agent.learn(replay_buffer, batch_size, gamma, tau, policy_noise, noise_clip)

            if done or t == (max_timesteps-1):
                break

        rewards.append(ep_reward)

        avg_reward = np.mean(rewards[-log_interval:])

        print("Episode: {}  \tReward: {:.3f}   \tAverage Reward: {:.3f}   \tTime Steps: {}"
              .format(episode, ep_reward, avg_reward, t))

        # logging updates:
        log_f.write('{},{}\n'.format(episode, ep_reward))
        log_f.flush()

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_ep = episode

        if ep_reward > 300:
            if test(env, agent):
                print("################################### Solved!! ###################################")
                name = filename + "_solved_in_" + str(episode)
                agent.save(directory, name)
                break

        if episode == max_episodes:
            name = filename + str(episode)
            agent.save(directory, name)

    n_episode = [i for i in range(episode)]
    fig = filename + ".png"
    plot(n_episode, rewards, filename=fig)
    end_time = time.time()
    training_time = end_time - start_time
    training_sec = training_time % 60
    training_min = (training_time / 60) % 60
    training_hr = (training_time / 60) / 60
    print("Training Time: {:.0f} hr {:.0f} min {:.0f} sec".format(training_hr, training_min, training_sec))
    print("Best episode: {}   \tBest episode reward: {:.3f}".format(best_ep, best_reward))


def test(env, agent):

    test_episodes = 100
    solved = False
    total_reward = 0

    print("Testing model", end='')

    for episodes in range(test_episodes):

        state = env.reset()
        done = False

        while not done:

            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward

        if episodes % 10 == 0:
            print(".", end='')

    avg_reward = total_reward / test_episodes
    print('\tTesting average reward : ', avg_reward)

    if avg_reward >= 300:
        solved = True

    return solved


if __name__ == '__main__':
    train()
