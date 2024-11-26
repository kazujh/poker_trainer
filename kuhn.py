import numpy as np

class CFR_Kuhn:
    cards = ["J", "Q", "K"]
    def __init__(self):
        self.playercard = ""
        self.oppcard = ""
        self.history = ""
        self.actions = ["p", "b"]
        self.node_map = {}
    
    def get_node(self, history: str, player: int):
        if history in self.node_map:
            return self.node_map[history]
        else:
            self.node_map[history] = Kuhn_Node(history, player)
            return self.node_map[history]

    def deal(self):
        self.playercard = np.random.choice(self.cards)

        self.oppcards = self.cards.copy()
        self.oppcards.remove(self.playercard)
        self.oppcard = np.random.choice(self.oppcards)
        

        #for debugging
        '''self.playercard = "K"
        self.oppcards = "J"'''
    
    def is_terminal(self, action_dict: str):
        if action_dict[-2:] == "pp" or action_dict[-2:] == "bb" or action_dict[-2:] == "bp":
            return True
        else:
            return False
    
    def cfr(self, history: str, player: int, reach_0, reach_1):
        #1 recursively run a single game tree
        #2 calculate reward when a terminal node is reached
        #3 calculate regret for each other terminal node by doing hypothetical rewards - actual rewards
        #4 keep track of probability of reaching each specific node (reach_prob)
        #5 update probabilities according to the chances of making the decision
        #6 update strategies after the iteration

        #use the action history to check if the node is terminal 
        if self.is_terminal(history):
            return self.showdown(history)
        
        #determine the reach probabilities from the perspective of the current node
        current_reach = reach_0 if player == 0 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0

        #get the current node and its strategy
        current_node = self.get_node(history, player)
        strategy = current_node.get_strategy(current_node.regrets)

        #add to the strategy sums by multiplying the strategy to the reach probability
        for i in range(len(self.actions)):
            if player == 0:
                current_node.strategySum[i] += reach_0 * strategy[i]
            else:
                current_node.strategySum[i] += reach_1 * strategy[i]

        #probability of reaching the next node = previous node's chance of picking the current node * current node's chance of picking the next node
        #so recursively, reach probability = node.strategy[action] * get_reach_probability(next_node)
        #can pass down the probability of picking the current node by using new_reach_0 and new_reach_1
        #use the reach probability and multiply it by the utility of the decision to determine the next strategy

        #move = np.random.choice(self.actions, p=strategy)
        

        utilities = np.zeros(len(self.actions))
        for i, action in enumerate(self.actions):
            if player == 0:
                new_reach_0 = reach_0 * strategy[i]
                new_reach_1 = reach_1
            else:
                new_reach_0 = reach_0
                new_reach_1 = reach_1 * strategy[i]
            new_history = history + action
            utilities[i] = -1 * self.cfr(new_history, 1 - player, new_reach_0, new_reach_1)
    
        #original method:
        #expected_value = 0.0
        #for prob, utility in zip(strategy, utilities):
        #    expected_value += prob * utility

        expected_value = np.sum(strategy * utilities)

        regrets = utilities - expected_value
        """print(f"regret for {current_node.history}: {regrets}")"""
        for i in range(len(self.actions)):
            current_node.regrets[i] += regrets[i] * opponent_reach
        

        return expected_value


    def showdown(self, history: str):
        if history[-2:] == "bp":
            return 1
        elif history[-2:] == "bb":
            if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                return 2 
            else:
                return -2
        elif history[-2:] == "pp":
            if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                return 1
            else:
                return -1


    
    def train(self, iterations=100):
        for i in range(iterations):
            self.deal()
            history = self.playercard
            self.cfr(history, 0, 1.0, 1.0)

        print("===== Strategies =====")
        for action, node in self.node_map.items():
            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")


class Kuhn_Node:
    def __init__(self, history, player, parent_node=None, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.history = history
        self.game = None
        self.card = history[:1]
        self.player = player
        self.parent_node = parent_node
        self.bet_node = None
        self.pass_node = None
        self.children = [self.pass_node, self.bet_node]
        
        self.regrets = np.zeros(self.NUM_ACTIONS)
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.actions = ["p", "b"]
        self.strategy = self.get_strategy(self.regretSum)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
    
    def get_strategy(self, regret_sum):
        strategy = np.maximum(regret_sum, 0)
        normalizing_sum = np.sum(strategy)
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

game = CFR_Kuhn()
game.train(iterations=1000)
    


            



