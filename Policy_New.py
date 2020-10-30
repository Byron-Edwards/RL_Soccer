import numpy as np
import time
import pickle
from games import Soccer, SoccerPLUS


class Policy:
    def __init__(self, game:Soccer, player_num):
        self.game = game
        self.player_num = player_num
        self.action_map = self.game.action_map
        self.offset_action = dict()
        for k, v in self.action_map.items():
            self.offset_action[tuple(v)] = k
        self.my_goal = self.game.getGoalLocation(player_num)
        self.opp_goal = self.game.getGoalLocation(not player_num)
        self.my_goal_up = self.my_goal[:3]
        self.my_goal_down = self.my_goal[-3:]
        self.update_status()
        self.policy_type = [self.policy0, self.policy1, self.policy2, self.policy3, self.policy4, self.policy5]

    def moving(self, closer: bool, player_number: bool, locations: np.ndarray, actions: np.ndarray, weight=(1, 1, 1)):
        assert locations.ndim == 2 and actions.ndim == 2
        current_location = self.me if player_number else self.opp
        current_dis = np.sum(abs(locations - current_location), axis=1)
        next_locations = current_location + actions
        min_dis, max_dis, mean_dis = [], [], []
        for next in next_locations:
            distances = locations - next
            distances = np.sum(abs(distances), axis=1)
            min_dis.append(np.min(distances))
            max_dis.append(np.max(distances))
            mean_dis.append(np.mean(distances))
        min_dis = np.array(min_dis) - np.min(current_dis)
        max_dis = np.array(max_dis) - np.max(current_dis)
        mean_dis = np.array(mean_dis) - np.mean(current_dis)
        scores = (weight[0] * min_dis + weight[1] * max_dis + weight[2] * mean_dis) * (1 if not closer else -1)
        scores[np.argwhere(scores < 0)] = 0
        return scores

    def validActionAll(self):
        all_actions = []
        current_location = self.me
        for i in range(len(self.action_map)):
            if hasattr(self.game, 'player_agent'):
                me = self.game.player_agent if self.player_num else self.game.player_opponent
                if me.energy_enough(i):
                    if self.game.getPlayerDistance() <= 4:
                        all_actions.append(self.action_map[i])
                    elif me.energy_cost[i] > 0:
                        all_actions.append(self.action_map[i])
            else:
                all_actions.append(self.action_map[i])
        all_actions = np.array(all_actions)
        new_locations = current_location + all_actions
        valid_actions = self.game.validLocationAll(new_locations)
        valid_new_locations = new_locations[valid_actions]
        all_actions = all_actions[valid_actions]
        return valid_new_locations, all_actions

    def update_status(self):
        self.has_ball = self.game.agent_keep_ball if self.player_num else not self.game.agent_keep_ball
        self.me = self.game.agent if self.player_num else self.game.opponent
        self.opp = self.game.opponent if self.player_num else self.game.agent

    def grab_ball(self, valid_actions):
        actions_score = self.moving(closer=True, player_number=False, locations=self.opp_goal, actions=valid_actions)
        action = self.score_to_index(valid_actions, actions_score)
        opp_new_location = self.opp + self.action_map[action]
        offset = opp_new_location - self.me
        if tuple(offset) in self.offset_action:
            return self.offset_action[tuple(offset)]
        else:
            return None

    def keep_ball(self):
        assert self.has_ball
        assert sum(abs(self.me - self.opp)) <= 1

    def win_the_game(self, valid_actions):
        current_location = self.me
        goal = self.my_goal
        new_locations = current_location + valid_actions
        for location, action in zip(new_locations,valid_actions):
            if (location == goal).all(1).any():
                return action
        return None

    def score_to_index(self, valid_actions, actions_score):
        action_index = []
        for offset in valid_actions:
            action_index.append(self.offset_action[tuple(offset)])
        if sum(actions_score) == 0:
            actions_prob = [1/len(actions_score) for i in range(len(actions_score))]
        else:
            actions_prob = actions_score / sum(actions_score)
        action = np.random.choice(action_index, p=actions_prob)
        return action

    # policy 0: if not has ball, get ball. if has ball get to goal
    def policy0(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    # policy 1: if not has ball, get ball. if has ball get to goal and keep away from opponent
    def policy1(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions,)
            if self.game.getPlayerDistance() <= 4:
                actions_score += self.moving(closer=False, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions, weight=(0.5, 0.5, 0.5))
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    # policy 2:
    # if not has ball, defence the goal if distance larger than 2. if less than 2 get closer try to grab the ball.
    # if has ball get to goal and keep away from opponent
    def policy2(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
            if self.game.getPlayerDistance() <= 4:
                actions_score += self.moving(closer=False, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions, weight=(0.5, 0.5, 0.5))
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp_goal, actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    # policy 3:
    # if not has ball, defence the goal if distance larger than 2. if less than 2 get closer try to grab the ball.
    # if has ball get to goal
    def policy3(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=self.opp_goal, actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    # policy 1: if not has ball, get ball. if has ball get to goal and keep away from opponent
    def policy4(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal_up, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    def policy5(self, valid_actions):
        if self.has_ball:
            offset = self.win_the_game(valid_actions)
            if offset is not None:
                return self.offset_action[tuple(offset)]
            actions_score = self.moving(closer=True, player_number=True, locations=self.my_goal_down, actions=valid_actions)
        else:
            actions_score = self.moving(closer=True, player_number=True, locations=np.expand_dims(self.opp, axis=0), actions=valid_actions)
            if self.game.getPlayerDistance() <= 2:
                action = self.grab_ball(valid_actions)
                if action is not None:
                    return action
        action = self.score_to_index(valid_actions, actions_score)
        return action

    def get_actions(self, policy_num):
        if isinstance(policy_num,str):
            policy_num = int(policy_num)
        self.update_status()
        if policy_num == -1:
            return np.random.randint(len(self.action_map))
        valid_new_locations, all_actions = self.validActionAll()
        return self.policy_type[policy_num](all_actions)


if __name__ == "__main__":
    policy_types = list(range(5))
    env = SoccerPLUS(visual=True)
    env.reset()
    my_policy = Policy(game=env, player_num=True)
    opp_policy = Policy(game=env, player_num=False)
    for rounds in range(1000):
        env.reset()
        while True:
            s_, reward, done, _ = env.step(my_policy.get_actions(5), opp_policy.get_actions(5))
            env.render()
            time.sleep(0.5)
            if done:
                print(reward)
                break
    # total_performance = dict()
    # win_rate = dict()
    # mean_scores = dict()
    # for my in range(4):
    #     for opp in range(4):
    #         policy_performance = list()
    #         for rounds in range(1000):
    #             env.reset()
    #             while True:
    #                 s_, reward, done, _ = env.step(my_policy.get_actions(my), opp_policy.get_actions(opp))
    #                 if done:
    #                     policy_performance.append(reward)
    #                     break
    #         total_performance["{}-{}".format(my, opp)] = policy_performance
    #         wins = np.sum(np.array(policy_performance) > 0) / 1000
    #         mean_score = np.mean(np.array(policy_performance))
    #         win_rate["{}-{}".format(my, opp)] = wins
    #         mean_scores["{}-{}".format(my, opp)] = mean_score
    #         print("MY Policy:\t{},\tOpp Policy:\t{},\tMeans Scores:\t{},\twin_rate:{}".format(my, opp, mean_score, wins))
    # with open("Policy_Performance", "wb") as f:
    #     pickle.dump((total_performance, win_rate, mean_scores,), f)
