import numpy as np

class CFR_Kuhn:
    cards = ["J", "Q", "K"]
    def __init__(self):
        self.playercard = ""
        self.oppcard = ""
        self.action_dict = ""
        self.actions = ["p", "b"]
        self.node_map = {}
    
    def get_node(self, action_dict: str, player: int):
        if action_dict in self.node_map:
            return self.node_map[action_dict]
        else:
            self.node_map[action_dict] = Kuhn_Node(action_dict, player)
            return self.node_map[action_dict]

    def deal(self):
        self.playercard = np.random.choice(self.cards)

        self.oppcards = self.cards.copy()
        self.oppcards.remove(self.playercard)
        self.oppcard = np.random.choice(self.oppcards) 
    
    def is_terminal(self, action_dict: str):
        if action_dict[-2:] == "pp" or action_dict[-2:] == "bb" or action_dict[-2:] == "bp":
            return True
        else:
            return False
    
    def cfr(self, action_dict: str, player: int, reach_0, reach_1):
        #1 recursively run a single game tree
        #2 calculate reward when a terminal node is reached
        #3 calculate regret for each other terminal node by doing hypothetical rewards - actual rewards
        #4 keep track of probability of reaching each specific node (reach_prob)
        #5 update probabilities according to the chances of making the decision
        #6 update strategies after the iteration
        if self.is_terminal(action_dict):
            print(f"final action_dict: {action_dict}")
            reward = self.showdown(action_dict)
            return reward
        current_player = player
        current_reach = reach_0 if current_player == 0 else reach_1
        opponent_reach = reach_1 if current_player == 0 else reach_0

        current_node = self.get_node(self.action_dict, player)

        strategy = current_node.get_strategy(current_node.regrets)


        move = current_node.get_move(current_node.strategy)
        print(move)        
        new_action_dict = action_dict + move # <--- should be a string
        print(f"action_dict: {new_action_dict}")

        if player == 0:
            new_reach_0 = reach_0 * strategy[self.actions.index(move)]
            new_reach_1 = reach_1
            current_node.reach_prob = new_reach_0
        else:
            new_reach_0 = reach_0
            new_reach_1 = reach_1 * strategy[self.actions.index(move)]
            current_node.reach_prob = new_reach_1

        
        for i in range(len(self.actions)):
            current_node[i] = -self.cfr(new_action_dict, 1 - player, new_reach_0, new_reach_1) # <--- need to update the reaches before recursively calling


        """current_node = self.get_node(self.action_dict)
        if current_node.is_terminal():
            reward_tup = self.showdown(self.action_dict)
            print(f"Reward_tupe: {reward_tup}")
            return np.array(self.calculate_regret(self.action_dict, reward_tup))
        else:
            next_node = self.move(player, self.opponent)
            print(f"Current player.regret_table: {player.regret_table}")
            player.regret_table[self.playercard] += np.flip(self.cfr(self.action_dict, next_node))
            player.strategy = player.get_strategy(player.regret_table)
            print(f"Current player.strategy: {player.strategy}")
            next_node.reach_prob = player.strategy[self.playercard][next_node.actions.index(next_node.action_dict[-1])]
            print(f"Next nodes reach probability: {next_node.reach_prob}")
            return player.regret_table[self.playercard]"""
        

    def train(self, iterations=100):
        for i in range(iterations):
            self.deal()
            self.action_dict = " "
            self.cfr(self.action_dict, self.player)
    
    def calculate_regret(self, action_dict: str, reward_tup: tuple):
        """actual_reward = self.calculate_reward(reward_tup)
        test_action, new_action_dict = action_dict[-1], action_dict[0:len(action_dict)-1]
        hypothetical_reward_tup = self.showdown(new_action_dict + self.actions[1 - self.actions.index(test_action)])
        hypothetical_reward = self.calculate_reward(hypothetical_reward_tup)
        
        if action_dict[-1:] == "p":
            return [actual_reward, hypothetical_reward - actual_reward]
        else:
            return [hypothetical_reward - actual_reward, actual_reward]"""
        #need to change this to be more efficient and well-written
        #adds to the current regret table for the node
        #still have not implemented traversing up the game tree to set the regret tables for the other nodes
        #this works for one " pbp" but need to bring it up the game tree to find the regrets for other nodes
        #need to pass this into the node so that it sets the values of the node's regrets to it


    def showdown(self, history: str):
        if history[-2:] == "bp":
            if len(history) == 4:
                return 1
            return -1
        else:
            if history[-2:] =="bb":
                if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                    return 2
                else:
                    return -2
            else:
                if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                    return 1
                else:
                    return -1

                    #STOPPED HERE
        
        



#bet node or pass node
class Kuhn_Node:
    def __init__(self, action_dict, player, parent_node=None, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.action_dict = action_dict
        self.game = None
        self.card = action_dict[:1]
        self.player = player
        self.parent_node = parent_node
        self.bet_node = None
        self.pass_node = None
        self.children = [self.pass_node, self.bet_node]
        
        self.regrets = np.zeros(self.NUM_ACTIONS)
        self.actions = ["p", "b"]
        self.strategy = self.get_strategy(self.regrets)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.reach_prob = 0.0
    
    def get_strategy(self, regret_sum):
        regret_sum[regret_sum < 0] = 0
        strategy = regret_sum.copy()
        normalizing_sum = sum(regret_sum)
        for a in range(self.NUM_ACTIONS):
            if normalizing_sum > 0:
                strategy[a] /= normalizing_sum
            else:
                strategy[a] = 1.0 / self.NUM_ACTIONS
        return strategy
    
    def get_average_strategy(self):
        if sum(self.strategySum) > 0:
            average_strategy = self.strategySum / sum(self.strategySum)
        else:
            average_strategy = np.full(self.NUM_ACTIONS, 1.0 / self.NUM_ACTIONS)
        return average_strategy
    
    def get_move(self, weights):
        return np.random.choice(self.actions, p=weights)
    
    def get_regrets(self):
        outcomes = self.game.showdown(self.action_dict)
        bet_regret = self.game.showdown(self.action_dict + "b")
        pass_regret = self.game.showdown(self.action_dict + "p")
        if self.player == 1:
            bet_regret = tuple(reversed(bet_regret))
            pass_regret = tuple(reversed(pass_regret))
        self.regret_table[self.card]["b"] += (bet_regret[0] - bet_regret[1]) - (pass_regret[0] - pass_regret[1])
        self.regret_table[self.card]["p"] += (pass_regret[0] - pass_regret[1]) - (bet_regret[0] - bet_regret[1])
        return self.regret_table
    




class Kuhn_Pass():
    def __init__(self):
        self.strategy = "p"
        self.regretSum = None
        self.game = None
    
    def get_move(self, weights):
        return self.strategy
    
    def get_strategy(self, regrets):
        return self.strategy
    
class Kuhn_Bet():
    def __init__(self):
        self.strategy = "b"
        self.regretSum = None
        self.game = None

    def get_move(self, weights):
        return self.strategy

    def get_strategy(self, regrets):
        return self.strategy

            



