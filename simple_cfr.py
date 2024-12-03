import numpy as np
import time

class CFRKuhn:
    def __init__(self):
        self.history = ""
        self.actions = ["p", "b"]
        self.n_actions = 2
        self.deck = [1, 2, 3]
        self.node_map = {}
    
    def get_node(self, history: str, playercard: int):
        key = str(playercard) + history
        if key in self.node_map:
            return self.node_map[key]
        else:
            self.node_map[key] = KuhnNode(history)
            return self.node_map[key]
    
    def is_terminal(self, history: str):
        return history[-2:] in ['bb', 'bp', 'pp']
    
    def cfr(self, history: str, player: int, reach_0, reach_1):
        #use the action history to check if the node is terminal
        playercard = self.deck[0] if player == 0 else self.deck[1]
        oppcard = self.deck[1] if player == 0 else self.deck[0]

        if self.is_terminal(history):
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


    
    def train_until_convergence(self, iterations=10000):
        nash_equilibrium = {
            "1 ": [.80, .20],
            "1 pb": [1.00, 0.00],
            "2 ": [1.00, 0.00],
            "2 pb": [.40, .60],
            "3 ": [.25, .75],
            "3 pb": [0.00, 1.00],
            "1 p": [.67, .33],
            "1 b": [1.00, 0.00],
            "2 p": [1.00, 0.00],
            "2 b": [.67, .33],
            "3 p": [0.00, 1.00],
            "3 b": [0.00, 1.00]
        }
        has_converged = False
        has_consistently_converged = False
        i = 0
        while not has_consistently_converged:
        #for i in range(iterations):
            np.random.shuffle(self.deck)
            history = " "
            self.cfr(history, 0, 1, 1)
            for _, node in self.node_map.items():
                node.update_strategy()
            
            if i % 10 == 0:
                converged = True
                for history, v in self.node_map.items():
                    if v.get_average_strategy()[0] - .05 < nash_equilibrium[history][0] and nash_equilibrium[history][0] < v.get_average_strategy()[0] + .05:
                        continue
                    else:
                        converged = False
                if converged:
                    if has_converged:
                        has_consistently_converged = True
                    else:
                        has_consistently_converged = False
                    has_converged = True
                    if has_consistently_converged:
                        print(f"Converged at iteration {i} with strategies:")
                        print("===== Player Strategies =====")
                        sorted_nodes = sorted(self.node_map.items())
                        for action, node in filter(lambda x: len(x[0]) % 2 == 0, sorted_nodes):
                            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")
                        
                        print()

                        print("===== Opponent Strategies =====")
                        for action, node in filter(lambda x: len(x[0]) % 2 == 1, sorted_nodes):
                            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")
                        
                        print()
                else:
                    if has_converged:
                        has_converged = False
            i += 1

    def train(self, iterations=10000):
        for i in range(iterations):
            np.random.shuffle(self.deck)
            history = " "
            self.cfr(history, 0, 1, 1)
            for _, node in self.node_map.items():
                node.update_strategy()
        
        print("Final Strategies:")
        print("===== Player Strategies =====")
        sorted_nodes = sorted(self.node_map.items())
        for action, node in filter(lambda x: len(x[0]) % 2 == 0, sorted_nodes):
            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")
        
        print()

        print("===== Opponent Strategies =====")
        for action, node in filter(lambda x: len(x[0]) % 2 == 1, sorted_nodes):
            print(f"{action} = [p: {node.get_average_strategy()[0]: .2f}, b: {node.get_average_strategy()[1]: .2f}]")
        
        print()


    def train_player(self):
        node_difficulties = {}
        for action_dict, node in self.node_map.items():
            node_difficulties[action_dict] = abs(round(node.get_average_strategy()[0] -  node.get_average_strategy()[1], 2))
        print(f"Node difficulties: {node_difficulties}")
        difficulties = list(set(node_difficulties.values()))
        difficulties.sort(reverse=True)
        print(f"difficulties: {difficulties}")


        moves = {"p": 0, "b": 1}
        playing = True

        correct = False
        #index of the difficulty
        difficulty_index = 0
        while playing:
            print(f"difficulty ind: {difficulty_index}")
            #value of the difficulty
            current_difficulty = difficulties[difficulty_index]
            print(f"current_difficulty: {current_difficulty}")
            #all action_dicts with that value
            games_with_difficulty = [key for key, value in node_difficulties.items() if value == current_difficulty]
            print(games_with_difficulty)


            game_state = np.random.choice(games_with_difficulty)
            print(game_state)
            current_node = self.node_map[game_state]
            current_strategies = current_node.get_average_strategy()
            print(f"Your card: {int(game_state[0])}")
            if len(game_state) > 3:
                print(f"You passed")
                print("Opponent bet")
            if len(game_state) == 3:
                if game_state[2] == 'b':
                    print(f"Opponent bet")
                else:
                    print(f"Opponent passed")
            move = str(input("What is your next move? (p/b)  "))
            if current_strategies[moves[move]] >= current_strategies[1 - moves[move]]:
                print("Correct!")
                correct = True
            else:
                print("incorrect")
                correct = False
            
            if correct:
                if difficulty_index < len(difficulties) - 1:
                    difficulty_index += 1
            else:
                if difficulty_index > 0:
                    difficulty_index -= 1
                #next have to implement giving the player a more/less difficult scenario based on the quality of their answer
                
            if str(input("Keep learning? (y/n)  ")).lower() == "n":
                print(f"final score: {difficulty_index}")
                playing = False

    
class KuhnNode:
    def __init__(self, history, parent_node=None, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.history = history
        
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
    time1 = time.time()
    game = CFRKuhn()
    game.train(iterations=100000)
    print(f"run time: {abs(time1 - time.time())}")
    #game.train_player()