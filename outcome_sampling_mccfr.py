import numpy as np
import time

class MCCFR:
    def __init__(self):
        self.history = ""
        self.actions = ["p", "b"]
        self.n_actions = 2
        self.deck = np.array([1, 2, 3])
        self.node_map = {}
        
        self.player = 0
        self.epsilon = 0.14

    def get_node(self, history: str, playercard: int):
        key = str(playercard) + history
        if key in self.node_map:
            return self.node_map[key]
        else:
            self.node_map[key] = KuhnNode(history)
            return self.node_map[key]
    
    def is_terminal(self, history: str):
        return history[-2:] in ['bb', 'bp', 'pp']
    
    def cfr(self, history: str, player: int, reach_0, reach_1, sampling_rate):
        #use the action history to check if the node is terminal
        playercard = self.deck[0] if player == 0 else self.deck[1]
        oppcard = self.deck[1] if player == 0 else self.deck[0]

        if self.is_terminal(history):
            reward = self.showdown(history, playercard, oppcard)
            return reward / sampling_rate, 1
        
        #get the current node and its strategy
        current_node = self.get_node(history, playercard)
        strategy = current_node.get_strategy()
        if player == self.player:
            probability = self.sample_strategy(strategy)
        else:
            probability = current_node.strategy

        move = current_node.get_move(probability)
        
        new_history = history + self.actions[move]
        if player == 0:
            util, p_tail = self.cfr(new_history, 1 - player, reach_0 * current_node.strategy[move], reach_1, sampling_rate * probability[move])
        else:
            util, p_tail = self.cfr(new_history, 1 - player, reach_0, reach_1 * current_node.strategy[move], sampling_rate * probability[move])
        util *= -1
        #determine the reach probabilities from the perspective of the current node
        current_reach = reach_0 if player == 1 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0
        if player == self.player:
            W = util * opponent_reach
            for a in range(len(strategy)):
                regret = W * (1.0 - strategy[move]) * p_tail if a == move else -W * strategy[move] * p_tail
                current_node.regretSum[a] += regret
        else:
            for a in range(len(strategy)):
                current_node.strategySum[a] += (current_reach * current_node.strategy[a]) / sampling_rate
        
        return util, p_tail * current_node.strategy[move]

    def sample_strategy(self, strategy):
        for i in range(len(strategy)):
            strategy[i] = (self.epsilon * np.repeat(1.0 / self.n_actions, self.n_actions)[i] + (1 - self.epsilon) * strategy[i])
        return strategy
    
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
            if i == iterations // 2:
                for _, v in self.node_map.items():
                    v.strategySum = np.zeros(v.NUM_ACTIONS)
            
            np.random.shuffle(self.deck)
            history = " "
            for j in range(2):
                self.player = j
                self.cfr(history, 0, 1, 1, 1)


                
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
    def __init__(self, history, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.history = history
        self.game = None
        
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.actions = ["p", "b"]
        self.moves = np.arange(self.NUM_ACTIONS)
        self.strategy = np.repeat(1/self.NUM_ACTIONS, self.NUM_ACTIONS)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.reach_pr = 0
        self.reach_pr_sum = 0

    def get_strategy(self):
        self.strategy = self.regretSum
        for i in range(len(self.regretSum)):
            if self.regretSum[i] < 0:
                self.strategy[i] = 0
        normalizing_sum = np.sum(self.strategy)
        if normalizing_sum > 0:
            self.strategy = self.strategy / normalizing_sum
        else:
            self.strategy = np.repeat(1 / self.NUM_ACTIONS, self.NUM_ACTIONS)
        return self.strategy
    
    def get_average_strategy(self):
        positive_regrets = np.maximum(self.regretSum, 0)
        normalizing_sum = np.sum(positive_regrets)
        if normalizing_sum > 0:
            return positive_regrets / normalizing_sum
        else:
            return np.full(self.NUM_ACTIONS, 1 / self.NUM_ACTIONS)
    
    def get_move(self, strategy):
        return np.random.choice(self.moves, p=strategy)

if __name__ == '__main__':
    time1 = time.time()
    game = MCCFR()
    game.train(iterations=100000)
    print(f"run time: {abs(time1 - time.time())}")
    #game.train_player()