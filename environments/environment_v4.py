import functools
import random
from copy import copy
import math

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from io import StringIO
from pettingzoo import ParallelEnv

class Agent:
    def __init__(self, x, y, label, action_mask):
        self.x = x
        self.y = y
        self.label = label
        self.action_mask = action_mask

class CustomActionMaskedEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, map_size, prisoner_num, guard_num):
        """The init method takes in environment arguments.

        Should define the following attributes:
        - escape x and y coordinates
        - guard x and y coordinates
        - prisoner x and y coordinates
        - timestamp
        - possible_agents

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.map_size = map_size
        self.escape_y = None
        self.escape_x = None
        self.guard1_y = None
        self.guard1_x = None
        self.guard2_y = None
        self.guard2_x = None
        self.guard3_y = None
        self.guard3_x = None
        self.prisoner1_y = None
        self.prisoner1_x = None
        self.prisoner2_y = None
        self.prisoner2_x = None
        self.prisoner3_y = None
        self.prisoner3_x = None
        self.timestep = None
        self.prisoner_num = prisoner_num
        self.guard_num = guard_num
        
        self.possible_agents = [f"prisoner{i}" for i in range(1, prisoner_num + 1)]
        self.guard_agents = [f"guard{i}" for i in range(1, guard_num + 1)]
        self.flag_guards = {a: False for a in self.guard_agents}
        self.attack_range = None
        self.attack_probability = 0.7
        
        self.prisoners = [Agent(0, self.map_size - 1, f"prisoner{i}", [0, 1, 0, 1]) for i in range(1, prisoner_num + 1)]
        self.guards = [Agent(random.randint(2, self.map_size - 1), random.randint(2, self.map_size - 1),
                             f"guard{i}", [0, 0, 0, 0]) for i in range(1, guard_num + 1)]


    def reset(self, seed=None, options=None):
        """Reset set the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - prisoner x and y coordinates
        - guard x and y coordinates
        - escape x and y coordinates
        - observation
        - infos

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        self.attack_range = 2
        
        for i, prisoner in enumerate(self.prisoners, start=1):
            setattr(prisoner, "x", 0)  #prisoner.x = 0
            setattr(prisoner, "y",  self.map_size - 1)

        for i, guard in enumerate(self.guards, start=1):
            setattr(guard, "x", random.randint(2, self.map_size - 1))
            setattr(guard, "y", random.randint(2, self.map_size - 1))

        self.escape_x = random.randint(2, self.map_size - 1)
        self.escape_y = random.randint(2, self.map_size - 1)
        
        self.terminations = {a: False for a in self.agents}
        
        # self.prisoners_obs = np.zeros((self.map_size, self.map_size), dtype=int)
        self.prisoners_obs = np.full((self.map_size, self.map_size), None, dtype=object)

        observations = self.get_observations()

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        """Takes in an action for the current agent (specified by agent_selection).

        Needs to update:
        - prisoner x and y coordinates
        - guard x and y coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """

        rewards = {a: 0 for a in self.agents}        
        # Execute actions
        self.update_agents(actions)
        observations = self.get_observations()

        print("term")
        print(self.terminations)
        
        # Generate action masks
        for prisoner in self.prisoners:
            prisoner.action_mask = np.ones(4, dtype=np.int8)

        # Update action masks based on prisoner positions
        for prisoner in self.prisoners:
            if prisoner.x == 0:
                prisoner.action_mask[0] = 0  # Block left movement
            elif prisoner.x == (self.map_size - 1):
                prisoner.action_mask[1] = 0  # Block right movement
            if prisoner.y == 0:
                prisoner.action_mask[2] = 0  # Block down movement
            elif prisoner.y == (self.map_size - 1):
                prisoner.action_mask[3] = 0  # Block up movement


        # Check termination conditions
        # terminations = {a: False for a in self.agents}
        # rewards = {a: 0 for a in self.agents}
        # def check_position(prisoner_x, prisoner_y, guard_x, guard_y, escape_x, escape_y):
        #     if prisoner_x == guard_x and prisoner_y == guard_y:
        #         return {"prisoners": -1, "guards": 1}
        #     elif prisoner_x == escape_x and prisoner_y == escape_y:
        #         return {"prisoners": 1, "guards": -1}
        #     else:
        #         return {}
        def check_position(prisoner_x, prisoner_y, guard_x, guard_y, escape_x, escape_y):
            result = {}
            
            if prisoner_x == guard_x and prisoner_y == guard_y:
                result["prisoners"] = result.get("prisoners", 0) - 1
                result["guards"] = result.get("guards", 0) + 1

            # Check if prisoners are at the escape point
            elif prisoner_x == escape_x and prisoner_y == escape_y:
                result["prisoners"] = result.get("prisoners", 0) + 1
                result["guards"] = result.get("guards", 0) - 1

            return result


        for prisoner in self.prisoners:
            for guard in self.guards:
                reward = check_position(prisoner.x, prisoner.y, guard.x, guard.y, self.escape_x, self.escape_y)
                rewards = {**reward}
        # if all(terminations.values()):
        #     break
        # if all(termination[a] for a in env.possible_agents):

        self.timestep += 1

        # Get observations
        observation = []
        
        for prisoner in self.prisoners:
            observation.append(prisoner.x + self.map_size * prisoner.y)
    
        for guard in self.guards:
            if self.flag_guards[guard.label]:
                observation.append(guard.x + self.map_size * guard.y)
            else:
                observation.append(-1)


        observation = tuple(observation)
        observation += (
            self.escape_x + self.map_size * self.escape_y,
            self.prisoners_obs
        )

        observations = {}
        
        # Update observations for prisoners
        for prisoner in self.prisoners:
            prisoner_observation_key = prisoner.label

            observations[prisoner_observation_key] = {
                "observation": observation,
                "action_mask": prisoner.action_mask
            }

        # Update observations for guards
        for guard in self.guards:
            guard_observation_key = guard.label

            observations[guard_observation_key] = {
                "observation": observation,
            }
#        print(observations)

        # Get dummy infos (not used in this example)
        infos = {"prisoner1": {}, "prisoner2": {}, "prisoner3": {}, "guard1": {}, "guard2": {}, "guard3": {}}

        return observations, rewards, self.terminations, infos

    def get_observations(self):
        observation = [p.x + self.map_size * p.y for p in self.prisoners]
        for g in self.guards:
            if self.flag_guards[g.label]:
                observation.append(g.x + self.map_size * g.y)
            else:
                observation.append(-1)

        observation.append(self.escape_x + self.map_size * self.escape_y)
        observation.append(self.prisoners_obs)
        return tuple(observation)

    def update_agents(self, actions): # agent들의 position 및 피습 check
        for prisoner, action in zip(self.prisoners, actions.values()):
            if not self.terminations[prisoner.label]:
                self.update_agent_position(prisoner, action)

        for guard in self.guards:
            for prisoner in self.prisoners:
                self.check_and_attack(guard, prisoner)

    def update_agent_position(self, agent, action): # action에 따른 agent 위치 이동
        if action == 0 and agent.x > 0:
            agent.x -= 1
        elif action == 1 and agent.x < (self.map_size - 1):
            agent.x += 1
        elif action == 2 and agent.y > 0:
            agent.y -= 1
        elif action == 3 and agent.y < (self.map_size - 1):
            agent.y += 1
    
    def check_attack_range(self, x1, y1, x2, y2): # 공격 범위 check
            squared_distance = (x1 - x2)**2 + (y1 - y2)**2
            distance = math.sqrt(squared_distance)
            return distance <= self.attack_range

    def check_and_attack(self, guard, prisoner): # guard가 공격 가능 범위 내 들어온 prisoner를 공격하는지 여부
        if self.check_attack_range(guard.x, guard.y, prisoner.x, prisoner.y):
            if random.random() < self.attack_probability:
                self.terminations[prisoner.label] = True
                self.prisoners_obs[guard.y, guard.x] = 1
                self.flag_guards[guard.label] = True # 공격 당하면 적의 위치를 알게 됨.
            else:
                self.prisoners_obs[prisoner.y, prisoner.x] = 0
        else:
            self.prisoners_obs[prisoner.y, prisoner.x] = 0



    def render(self):
        """Renders the environment."""
    
        grid = np.full((self.map_size, self.map_size), '', dtype=object)

        grid[self.escape_y, self.escape_x] = "E"

        guards = [{"x": guard.x, "y": guard.y, "label_G": f"G{i}"} for i, guard in enumerate(self.guards, start=1)]
        for guard in guards:
            grid[guard["y"], guard["x"]] += guard["label_G"]

        prisoners = [{"x": prisoner.x, "y": prisoner.y, "label_P": f"P{i}"} for i, prisoner in enumerate(self.prisoners, start=1)]
        for prisoner in prisoners:
            grid[prisoner["y"], prisoner["x"]] += prisoner["label_P"]

        output = StringIO()
    
        print("+" + "-" * (self.map_size * 4 - 1) + "+", file=output)
    
        for row in grid:
            print("|", end='', file=output)
            for cell in row:
                print(f"{cell:^3}", end='', file=output)
                print("|", end='', file=output)
            print("\n+" + "-" * (self.map_size * 4 - 1) + "+", file=output)

        output.seek(0)

        # Print the formatted output
        print(output.read())



    # def render(self):
    #     """Renders the environment."""
        
    #     grid = np.full((self.map_size, self.map_size), '', dtype=object)

    #     grid[self.escape_y, self.escape_x] = "E"

    #     guards = [{"x": guard.x, "y": guard.y, "label_G": f"G{i}"} for i, guard in enumerate(self.guards, start=1)]

    #     for guard in guards:
    #         grid[guard["y"], guard["x"]] += guard["label_G"]

    #     prisoners = [{"x": prisoner.x, "y": prisoner.y, "label_P": f"P{i}"} for i, prisoner in enumerate(self.prisoners, start=1)]

    #     for prisoner in prisoners:
    #         grid[prisoner["y"], prisoner["x"]] += prisoner["label_P"]
        
    #     for row in grid:
    #         print(row)
        # print('\n')
        # print(self.prisoners_obs)


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([self.map_size * self.map_size - 1] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)