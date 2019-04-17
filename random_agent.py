#!/usr/bin/env python

import retro
import gym
from action_space import make_custom_env


def main():
    # Default Discrete Action Spaces => results in 36 different possible actions
    env = retro.make(game='SuperMarioBros3-Nes',
                     use_restricted_actions=retro.Actions.DISCRETE)
    print('retro.Actions.DISCRETE action_space', env.action_space)
    env.close()

    # Multibinary Action Space =>
    env = make_custom_env(disc_acts=False)
    print('MarioCustomDiscretizer action_space', env.action_space)
    env.close()

    # Custom Discrete Action Space => results in only 7 discrete actions
    env = make_custom_env(disc_acts=True)
    print('MarioCustomDiscretizer action_space', env.action_space)

    # reset environment and get initial data
    obs = env.reset()
    init_data = env.data.lookup_all()
    cur_nr_lives = int(init_data['lives'])

    print("Initial State of Mario:", init_data)

    while True:
        # choose a random action
        rand_action = env.action_space.sample()

        # perform the random action
        obs, rew, done, info = env.step(rand_action)
        env.render()

        if done or int(info['lives']) < cur_nr_lives or int(info['time']) == 0:
            # update the number of remaining lives
            print(done, info)
            cur_nr_lives = int(info['lives'])
            print(cur_nr_lives)

            # reset the environment
            obs = env.reset()

            print("Mario died or time ran out!")
            print("Mario's Current State:", env.data.lookup_all())

            # Mario is Gameover
            if cur_nr_lives == 0:
                break

    env.render(close=True)
    env.close()


if __name__ == "__main__":
    main()
