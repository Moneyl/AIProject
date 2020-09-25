import gym

# Load an environment
env = gym.make('MountainCar-v0')

# Run 20 episodes. Resets environment at start of each run
for i_episode in range(20):
    observation = env.reset()

    # For each episode step 100 times
    for t in range(100):
        # Render the graphics/text/etc of the environment
        env.render()
        print(observation)

        # Get action that the agent/AI chose this step (I think)
        action = env.action_space.sample()
        # Step the simulation and get info about the environment and AI following that step
        observation, reward, done, info = env.step(action)
        
        # Once episode is done print number of timesteps and break from inner for loop
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

# After all episodes have run, close the environment
env.close()