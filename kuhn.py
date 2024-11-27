import numpy as np

class CFR_Kuhn:
    def __init__(self):
        self.history = ""
        self.actions = ["p", "b"]
        self.n_actions = 2
        self.deck = [1, 2, 3]
        self.cards = ["J", "Q", "K"]
        self.cards2 = self.cards.copy()
        self.node_map = {}
    
    def get_node(self, history: str, player: int):
        key = str(player) + history
        if key in self.node_map:
            return self.node_map[key]
        else:
            self.node_map[key] = Kuhn_Node(history)
            return self.node_map[key]
    
    def is_terminal(self, history: str):
        return history[-2:] in ['bb', 'bp', 'pp']
    
    def cfr(self, history: str, player: int, reach_0, reach_1):
        #use the action history to check if the node is terminal
        playercard = self.deck[0] if player == 0 else self.deck[1]
        oppcard = self.deck[1] if player == 0 else self.deck[0]

        #if using non-int values for the cards
        """playercard = self.cards[0] if player == 0 else self.cards[1]
        oppcard = self.cards[1] if player == 0 else self.cards [0] """
        if self.is_terminal(history):
            #if using non-int values for the cards
            """player_val = self.cards2.index(playercard)
            opp_val = self.cards2.index(oppcard)"""
            #changed the parameters as well
            return self.showdown(history, playercard, oppcard)
        
        #determine the reach probabilities from the perspective of the current node
        current_reach = reach_0 if player == 0 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0

        #get the current node and its strategy
        current_node = self.get_node(history, playercard)
        strategy = current_node.strategy
        
        utilities = np.zeros(self.n_actions)
        for i in range(self.n_actions):
            new_history = history + self.actions[i]
            if player == 0:
                utilities[i] = -1 * self.cfr(new_history, 1 - player, reach_0 * strategy[i], reach_1)
            else:
                utilities[i] = -1 * self.cfr(new_history, 1 - player, reach_0, reach_1 * strategy[i])

        node_util = sum(strategy * utilities)

        regrets = utilities - node_util

        current_node.reach_pr += current_reach
        current_node.regretSum += regrets * opponent_reach
        
        return node_util


    def showdown(self, history: str, player_val, opp_val):
        end_pass = history[-1] == "p"
        double_bet = history[-2:] == "bb"
        if end_pass:
            if history[-2:] == "pp":
                return 1 if player_val > opp_val else -1
            else:
                return 1
        elif double_bet:
            return 2 if player_val > opp_val else -2


    
    def train(self, iterations=100):
        for i in range(iterations):
            np.random.shuffle(self.deck)
            np.random.shuffle(self.cards)
            history = " "
            self.cfr(history, 0, 1, 1)
            for _, v in self.node_map.items():
                v.update_strategy()

                

        print("===== Player Strategies =====")
        sorted_nodes = sorted(self.node_map.items())
        for action, node in filter(lambda x: len(x[0]) % 2 == 0, sorted_nodes):
            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")
        
        print()

        print("===== Opponent Strategies =====")
        for action, node in filter(lambda x: len(x[0]) % 2 == 1, sorted_nodes):
            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")

    def train_player(self):
        difficulties = {}
        for action_dict, node in self.nodeMap.items():
            difficulties[action_dict] = abs(round(node.strategy[0], 2) -  round(node.strategy[1], 2))
        sorted_difficulties = sorted(difficulties.items(), key=lambda x: x[1], reverse=True)
        difficulties = {key: value for key, value in filter(lambda d: len(d[0]) % 2 == 0, sorted_difficulties)}


        current_difficulty = 1 - max(difficulties.values())
        games_with_difficulty = [key for key, value in difficulties.items() if value == 1 - current_difficulty]
        print(games_with_difficulty)
        playing = True

        cards = ["J", "Q", "K"]
        moves = {"p": 0, "b": 1}

        correct = False
        while playing:
            game_state = np.random.choice(games_with_difficulty)
            current_node = self.nodeMap[game_state]
            current_strategies = current_node.strategy
            print(f"Your card: {cards[int(game_state[0])]}")
            if len(game_state) > 2:
                print(f"You passed")
                print("Opponent bet")
            move = str(input("What is your next move? (p/b)  "))
            if current_strategies[moves[move]] >= current_strategies[1 - moves[move]]:
                print("Correct!")
                correct = True
            else:
                print("incorrect")
                correct = False
            if correct:
                ...
                #next have to implement giving the player a more/less difficult scenario based on the quality of their answer
                
            if str(input("Keep learning? (y/n)  ")).lower() == "n":
                playing = False

    
class Kuhn_Node:
    def __init__(self, history, parent_node=None, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.history = history
        self.game = None
        self.card = history[:1]
        self.parent_node = parent_node
        self.bet_node = None
        self.pass_node = None
        self.children = [self.pass_node, self.bet_node]
        
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.actions = ["p", "b"]
        self.strategy = np.repeat(1/self.NUM_ACTIONS, self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.reach_pr = 0
        self.reach_pr_sum = 0
    
    def update_strategy(self):
        self.strategySum += self.reach_pr * self.strategy
        self.reach_pr_sum += self.reach_pr
        self.strategy = self.get_strategy(self.regretSum)
        self.reach_pr = 0

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

if __name__ == '__main__':
    game = CFR_Kuhn()
    game.train(iterations=10000)