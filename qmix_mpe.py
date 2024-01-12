from PettingZoo.pettingzoo.mpe import simple_adversary_v3

env = simple_adversary_v3.env(render_mode='human')

env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample() # this is where you would insert your policy
    print(action)
    env.step(action)
env.close()