import copy
import hashlib
import math
import sys
import time
import random
import numpy as np
import copy
import queue

from game_env import GameEnv
from game_state import GameState

"""
solution.py

Template file for you to implement your solution to Assignment 3.

You must implement the following method stubs, which will be invoked by the simulator during testing:
    __init__(game_env)
    run_training()
    select_action()
    
To ensure compatibility with the autograder, please avoid using try-except blocks for Exception or OSError exception
types. Try-except blocks with concrete exception types other than OSError (e.g. try: ... except ValueError) are allowed.

COMP3702 2021 Assignment 3 Support Code

Last updated by njc 10/10/21

Student: Jingbo Ma

"""


class RLAgent:

    def __init__(self, game_env):
        """
        Constructor for your solver class.

        Any additional instance variables you require can be initialised here.

        Computationally expensive operations should not be included in the constructor, and should be placed in the
        plan_offline() method instead.

        This method has an allowed run time of 1 second, and will be terminated by the simulator if not completed within
        the limit.
        """
        self.game_env = game_env

        #
        #
        # TODO: Initialise any instance variables you require here.
        #
        #
        self.ACTIONS = [GameEnv.WALK_LEFT, GameEnv.WALK_RIGHT, GameEnv.JUMP, GameEnv.GLIDE_LEFT_1,
                        GameEnv.GLIDE_LEFT_2, GameEnv.GLIDE_LEFT_3, GameEnv.GLIDE_RIGHT_1, GameEnv.GLIDE_RIGHT_2,
                        GameEnv.GLIDE_RIGHT_3, GameEnv.DROP_1, GameEnv.DROP_2, GameEnv.DROP_3]
        self.q_values = {}  # dict mapping (state, action) to q values
        self.n_s = {}  # dict mapping states to its visit counts
        self.best_actions = {}
        self.n_sa = {}  # dict mapping (state, action) to counts
        self.reachable_states = []
        self.valid_actions = {}  # position (row, col) -> list(valid actions from this position)
        self.action_validity = {}

        self.iteration_counter = 0
        self.discount = 0.9995
        self.ALPHA = 0.02
        self.c = 1.5
        self.iterations = 300

        self.use_sarsa = False
        self.num_of_episodes = 0
        self.reward = [0]

        reachable_positions, self.valid_actions = self.__get_reachable_positions_and_valid_actions()
        reachable_statuses = self.__get_reachable_gem_statuses()
        for pos in reachable_positions:
            for status in reachable_statuses:
                self.reachable_states.append(GameState(*pos, status))
        self.persistent_state = random.choice(self.reachable_states)
        for state in self.reachable_states:
            self.n_s[state] = 0
            for action in self.valid_actions[(state.row, state.col)]:
                self.q_values[(state, action)] = 0
                self.n_sa[(state, action)] = 0

        self.sarsa_action = random.choice(self.valid_actions[(self.persistent_state.row, self.persistent_state.col)])

    def run_training(self):
        """
        This method will be called once at the beginning of each episode.

        You can use this method to perform training (e.g. via Q-Learning or SARSA).

        The allowed run time for this method is given by 'game_env.training_time'. The method will be terminated by the
        simulator if it does not complete within this limit - you should design your algorithm to ensure this method
        exits before the time limit is exceeded.

        Credit: Partially derived from A2 walk through video
        """
        t0 = time.time()

        #
        #
        # TODO: Code for training can go here
        #
        #
        while self.game_env.get_total_reward() > self.game_env.training_reward_tgt and \
                time.time() - t0 < self.game_env.training_time - 1:

            if self.use_sarsa:
                episode_end = self.__sarsa_iteration(self.game_env)
            else:
                episode_end = self.__q_learning_iteration(self.game_env)
            self.iteration_counter = self.iteration_counter + 1
            if episode_end or self.iteration_counter > self.iterations:
                self.num_of_episodes += 1
                self.reward.append(0)
                self.iteration_counter = 0
                new_state = self.persistent_state
                while new_state == self.persistent_state:
                    random.seed(time.time())
                    new_state = random.choice(self.reachable_states)
                self.persistent_state = new_state
                self.sarsa_action = random.choice(self.valid_actions[(new_state.row, new_state.col)])
        print("Episode count: ", self.num_of_episodes)
        title = ""
        if self.use_sarsa:
            title += "SARSA_"
        else:
            title += "Q_"
        title += "rewards.txt"
        f = open(title, "w")
        f.write("a=" + str(self.ALPHA) + "\n")
        for i in range(len(self.reward) - 1):
            f.write(str(i) + "," + str(self.reward[i]) + "\n")
        f.close()

        # optional: loop for ensuring your code exits before exceeding the reward target or time limit

            #
            #
            # TODO: Code for training can go here
            #
            #

    @staticmethod
    def calculate_ucb1(q_value, c, big_n, n_a):
        return q_value + c * np.sqrt(np.log(big_n) / n_a)

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def select_action(self, state):
        """
        This method will be called each time the agent is called upon to decide which action to perform (once for each
        step of the episode).

        You can use this method to select an action based on the Q-value estimates learned during training.

        The allowed run time for this method is 1 second. The method will be terminated by the simulator if it does not
        complete within this limit - you should design your algorithm to ensure this method exits before the time limit
        is exceeded.

        :param state: the current state, a GameState instance
        :return: action, the selected action to be performed for the current state
        """

        #
        #
        # TODO: Code for selecting an action based on learned Q-values can go here
        #
        #
        # choose the action with the highest Q-value for the given state
        best_q = -math.inf
        best_a = None

        for action in self.valid_actions[(state.row, state.col)]:
            if (state, action) in self.q_values.keys():
                q_value = self.q_values[(state, action)]
                if q_value > best_q:
                    best_q = q_value
                    best_a = action

        if best_a is None or best_q == 0:
            random.seed(time.time())
            return random.choice(self.valid_actions[(state.row, state.col)])
        else:
            return best_a

    #
    #
    # TODO: Code for any additional methods you need can go here
    #
    #
    @staticmethod
    def __stable_hash(x):
        return hashlib.md5(str(x).encode('utf-8')).hexdigest()

    def select_action_ucb(self, state):
        best_q = -math.inf
        best_ucb1 = -math.inf
        best_a = None
        unvisited = []
        q_value = None
        for action in self.valid_actions[(state.row, state.col)]:
            if (state, action) in self.q_values.keys():
                q_value = self.q_values[(state, action)]
                big_n = self.n_s[state]
                n_a = self.n_sa[(state, action)]
                if n_a == 0:
                    unvisited.append(action)
                    continue
                this_ucb1 = self.calculate_ucb1(q_value, self.c, big_n, n_a)
                if this_ucb1 > best_ucb1:
                    best_ucb1 = this_ucb1
                    best_a = action
        if len(unvisited) > 0:
            random.seed(time.time())
            return random.choice(unvisited)
            # If n_a == 0 for at least one action,
            # then choose one of the unvisited actions at random
        if best_a is None or q_value == 0:
            random.seed(time.time())
            return random.choice(self.valid_actions[(state.row, state.col)])
        else:
            return best_a

    def __action_is_valid(self, state, action):
        """
        Credit: A2 walk through video
        """
        if (state.row, state.col, action) in self.action_validity:
            return self.action_validity[(state.row, state.col, action)]

        if (action in GameEnv.WALK_AND_JUMP_ACTIONS and
            self.game_env.grid_data[state.row + 1][state.col] not in GameEnv.WALK_JUMP_ALLOWED_TILES) or \
                (action not in GameEnv.WALK_AND_JUMP_ACTIONS and
                 self.game_env.grid_data[state.row + 1][state.col] not in GameEnv.GLIDE_DROP_ALLOWED_TILES):
            self.action_validity[(state.row, state.col, action)] = False
            return False

        self.action_validity[(state.row, state.col, action)] = True
        return True

    def __get_reachable_positions_and_valid_actions(self):
        """
        Credit: Modified from A2 walk through video
        """
        init_pos = self.game_env.gem_positions[0]
        # init_pos = (self.game_env.get_init_state().row, self.game_env.get_init_state().col)
        status = tuple(0 for g in self.game_env.gem_positions)

        container = [init_pos]
        visited = {init_pos}
        valid_actions = {}
        while len(container) > 0:
            pos = container.pop(-1)
            s = GameState(*pos, status)

            va = [a for a in self.game_env.ACTIONS if self.__action_is_valid(s, a)]
            for a in va:
                outcomes = [self.game_env.perform_action(s.deepcopy(), a)]
                for o in outcomes:
                    n_pos = (o[2].row, o[2].col)
                    if n_pos not in visited:
                        container.append(n_pos)
                        visited.add(n_pos)
            if pos not in valid_actions.keys():
                valid_actions[pos] = va
        return visited, valid_actions

    def __get_reachable_gem_statuses(self):
        """
        Credit: A2 walk through video
        """
        container = [(0,), (1,)]
        statuses = []
        while len(container) > 0:
            cur = container.pop(-1)
            if len(cur) < self.game_env.n_gems:
                container.append((*cur, 0))
                container.append((*cur, 1))
            else:
                statuses.append(cur)
        return statuses

    def __q_learning_iteration(self, game_env):
        """
        Credit: Modified from Tutorial 10 solution
        """

        if self.iteration_counter > self.iterations:
            self.iteration_counter = 0
            return True

        #action = self.select_action(self.persistent_state)
        action = self.select_action_ucb(self.persistent_state)

        # update counter
        if self.persistent_state not in self.n_s.keys():
            self.n_s[self.persistent_state] = 1
        else:
            self.n_s[self.persistent_state] += 1

        if (self.persistent_state, action) not in self.n_sa.keys():
            self.n_sa[(self.persistent_state, action)] = 1
        else:
            self.n_sa[(self.persistent_state, action)] += 1

        # ===== simulate result of action =====
        action_is_valid, received_reward, next_state, state_is_terminal = \
            game_env.perform_action(self.persistent_state.deepcopy(), action)
        if not action_is_valid:
            raise ValueError("Invalid action!!")

        self.reward[self.num_of_episodes] += received_reward

        # ===== update value table =====
        best_q = -math.inf
        best_action = None

        if next_state not in self.reachable_states:    # Add the valid but not initialised state to the table
            self.reachable_states.append(next_state)
            self.n_s[next_state] = 1
            self.valid_actions[(next_state.row, next_state.col)] = \
                [a for a in self.game_env.ACTIONS if self.__action_is_valid(next_state, a)]

            for try_action in self.valid_actions[(next_state.row, next_state.col)]:
                self.q_values[(next_state, try_action)] = 0
                self.n_sa[(next_state, try_action)] = 0

        for action_dash in self.valid_actions[(next_state.row, next_state.col)]:
            if self.q_values[(next_state, action_dash)] > best_q:
                best_q = self.q_values[(next_state, action_dash)]
                best_action = action_dash
        #if best_action is None or state_is_terminal:
        if best_action is None:
            best_q = 0
        target = received_reward + self.discount * best_q
        if (self.persistent_state, action) in self.q_values:
            old_q = self.q_values[(self.persistent_state, action)]
        else:
            old_q = 0
        self.q_values[(self.persistent_state, action)] = old_q + self.ALPHA * (target - old_q)

        if state_is_terminal:
            self.iteration_counter = 0
            return True
        else:
            self.persistent_state = next_state
            self.iteration_counter += 1
            return False

    def __sarsa_iteration(self, game_env):
        """
        Credit: Modified from Tutorial 10 solution
        """

        if self.iteration_counter > self.iterations:
            self.iteration_counter = 0
            return True

        # update counter
        if self.persistent_state not in self.n_s.keys():
            self.n_s[self.persistent_state] = 1
        else:
            self.n_s[self.persistent_state] += 1

        if (self.persistent_state, self.sarsa_action) not in self.n_sa.keys():
            self.n_sa[(self.persistent_state, self.sarsa_action)] = 1
        else:
            self.n_sa[(self.persistent_state, self.sarsa_action)] += 1

        # ===== simulate result of action =====
        action_is_valid, received_reward, next_state, state_is_terminal = \
            game_env.perform_action(self.persistent_state.deepcopy(), self.sarsa_action)
        if not action_is_valid:
            raise ValueError("Invalid action!!")

        self.reward[self.num_of_episodes] += received_reward

        # ===== update value table =====
        if next_state not in self.reachable_states:    # Add the valid but not initialised state to the table
            self.reachable_states.append(next_state)
            self.n_s[next_state] = 1
            self.valid_actions[(next_state.row, next_state.col)] = \
                [a for a in self.game_env.ACTIONS if self.__action_is_valid(next_state, a)]

            for try_action in self.valid_actions[(next_state.row, next_state.col)]:
                self.q_values[(next_state, try_action)] = 0
                self.n_sa[(next_state, try_action)] = 0

        next_action = self.select_action_ucb(next_state)
        new_q = 0
        if (next_state, next_action) in self.q_values.keys():
            new_q = self.q_values[(next_state, next_action)]

        if state_is_terminal:
            new_q = 0

        target = received_reward + self.discount * new_q
        if (self.persistent_state, self.sarsa_action) in self.q_values.keys():
            old_q = self.q_values[(self.persistent_state, self.sarsa_action)]
        else:
            old_q = 0
        self.q_values[(self.persistent_state, self.sarsa_action)] = old_q + self.ALPHA * (target - old_q)

        if state_is_terminal:
            self.iteration_counter = 0
            return True
        else:
            self.persistent_state = next_state
            self.sarsa_action = next_action
            self.iteration_counter += 1
            return False
