from environments.custom_environment_v0 import CustomActionMaskedEnvironment

if __name__ == "__main__":
    env = CustomActionMaskedEnvironment()
    max_episode = 1
    max_step = 10
    
    for i in range(max_episode):

        observation, info = env.reset()
        for step in range(max_step):
            actions = {a: env.action_space(a).sample() for a in env.possible_agents}
            observation, reward, termination, info = env.step(actions)

            # Termination conditions
            if all(termination[a] for a in env.possible_agents):
                break
            
            # Rendering
            env.render()

    env.close