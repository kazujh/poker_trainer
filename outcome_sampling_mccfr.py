import numpy as np
import time

class MCCFR:
    def __init__(self):
        self.history = ""
        self.actions = ["p", "b"]
        self.n_actions = 2
        self.deck = np.array([0, 1, 2])
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

    def update_epsilon(self, iteration, max_iterations):
        epsilon_start = .14
        epsilon_end = .01
        self.epsilon = epsilon_start - iteration * (epsilon_start - epsilon_end) / max_iterations

    def cfr(self, history: str, player: int, reach_0, reach_1, sampling_rate, weight):
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
            util, p_tail = self.cfr(new_history, 1 - player, reach_0 * current_node.strategy[move], reach_1, sampling_rate * probability[move], weight)
        else:
            util, p_tail = self.cfr(new_history, 1 - player, reach_0, reach_1 * current_node.strategy[move], sampling_rate * probability[move], weight)
        util *= -1
        current_reach = reach_0 if player == 1 else reach_1
        opponent_reach = reach_1 if player == 0 else reach_0
        if player == self.player:
            W = util * opponent_reach
            for a in range(len(strategy)):
                regret = W * (1.0 - strategy[move]) * p_tail if a == move else -W * strategy[move] * p_tail
                current_node.regretSum[a] += regret
        else:
            for a in range(len(strategy)):
                current_node.strategySum[a] += (weight * current_reach * current_node.strategy[a]) / sampling_rate
        
        return util, p_tail * current_node.strategy[move]

    def calculate_exploitability(self, iterations=10000):
        best_response_p1 = self.best_response(0, iterations)
        best_response_p2 = self.best_response(1, iterations)
        return (best_response_p1 + best_response_p2) / 2

    def best_response(self, player, iterations):

        payoff = 0
        for i in range(iterations):
            np.random.shuffle(self.deck)
            player_card = self.deck[0] if player == 0 else self.deck[1]
            opp_card = self.deck[1] if player == 0 else self.deck[0]
            payoff += self.best_response_utility(" ", player, player_card, opp_card)
        avg_payoff = payoff / iterations
        return avg_payoff
        
    def best_response_utility(self, history, player, player_card, opp_card):
        if self.is_terminal(history):
            return self.showdown(history, player_card, opp_card)
    
        current_player = (len(history) - 1) % 2
        
        if current_player == player:
            best_utility = float("-inf")

            for a in self.actions:
                new_history = history + a
                utility = self.best_response_utility(new_history, player, player_card, opp_card)
                best_utility = max(utility, best_utility)
            return best_utility
        else:
            node = self.get_node(history, opp_card)
            strategy = node.get_average_strategy()

            utility = 0
            for i, a in enumerate(self.actions):
                new_history = history + a
                utility += strategy[i] * self.best_response_utility(new_history, player, player_card, opp_card)
            return utility
    
    def get_worst_node(self, nodes_dict: dict, nodes: list):
        values = np.zeros(len(nodes))
        for i, key in enumerate(nodes):
            values[i] = nodes_dict[key]
        normalizing_sum = sum(values)
        probabilities = np.zeros(len(nodes))
        for i in range(len(nodes)):
            probabilities[i] = values[i] / normalizing_sum
        return probabilities



    def train_until_convergence(self, iterations=10000):
        nash_equilibrium = {
            "0 ": [.80, .20],
            "0 pb": [1.00, 0.00],
            "1 ": [1.00, 0.00],
            "1 pb": [.40, .60],
            "2 ": [.25, .75],
            "2 pb": [0.00, 1.00],
            "0 p": [.67, .33],
            "0 b": [1.00, 0.00],
            "1 p": [1.00, 0.00],
            "1 b": [.67, .33],
            "2 p": [0.00, 1.00],
            "2 b": [0.00, 1.00]
        }
        has_converged = False
        has_consistently_converged = False
        i = 0
        while not has_consistently_converged:
        #for i in range(iterations):
            if i == iterations // 2:
                for _, v in self.node_map.items():
                    v.strategySum = np.zeros(v.NUM_ACTIONS)
            
            if i >= iterations // 2:
                self.update_epsilon(i, iterations)

            np.random.shuffle(self.deck)
            history = " "
            for j in range(2):
                self.player = j
                self.cfr(history, 0, 1, 1, 1)
            if i % 10 == 0:
                converged = True
                for history, v in self.node_map.items():
                    if v.get_average_strategy()[0] - .1 < nash_equilibrium[history][0] and nash_equilibrium[history][0] < v.get_average_strategy()[0] + .1:
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
            if i == iterations // 2:
                for _, v in self.node_map.items():
                    v.strategySum = np.zeros(v.NUM_ACTIONS)
            
            if i >= iterations // 4:
                self.update_epsilon(i, iterations)

            np.random.shuffle(self.deck)
            history = " "
            weight = i + 1
            for j in range(2):
                self.player = j
                self.cfr(history, 0, 1, 1, 1, weight) 
        exploitability = self.calculate_exploitability(iterations // 10)

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

        print(f"epsilon: {self.epsilon}")
        print(f"exploitability: {exploitability}")


    def train_player(self):
        node_difficulties = {}
        for action_dict, node in self.node_map.items():
            node_difficulties[action_dict] = abs(round(node.get_average_strategy()[0] -  node.get_average_strategy()[1], 2))
        difficulties = list(node_difficulties.values())
        for i, difficulty in enumerate(difficulties):
            difficulties[i] = float(str(difficulty)[:3])
        difficulties = list(set(difficulties))
        difficulties.sort(reverse=True)


        moves = {"p": 0, "b": 1}
        playing = True

        correct = False
        #index of the difficulty
        nodeAttempts = {key: 5.0 for key in node_difficulties.keys()}
        difficulty_index = (len(difficulties) - 1) // 4
        while playing:
            #value of the difficulty
            current_difficulty = difficulties[difficulty_index]
            print(f"current_difficulty: {difficulty_index}")
            #all action_dicts with that value
            games_with_difficulty = [key for key, value in node_difficulties.items() if (value < current_difficulty + .1 and value >= current_difficulty)]

            weights = self.get_worst_node(nodeAttempts, games_with_difficulty)
            game_state = np.random.choice(games_with_difficulty, p=weights)
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
            while move != 'p' and move != 'b':
                move = str(input("Please make a valid move (p/b)  "))
            if current_strategies[moves[move]] + .05 >= current_strategies[1 - moves[move]] - .05:
                print("Correct!")
                correct = True
            else:
                print("incorrect")
                correct = False

            positive_increment = (len(difficulties) - 1 - difficulty_index) // 2
            negative_increment = difficulty_index // 2
            if correct:
                if difficulty_index < len(difficulties) - 1:
                    difficulty_index += positive_increment if positive_increment != 0 else 1
                if nodeAttempts[game_state] > 1:
                    nodeAttempts[game_state] -= 1
            else:
                if difficulty_index > 0:
                    difficulty_index -= negative_increment if negative_increment != 0 else 1
                if nodeAttempts[game_state] < 5:
                    nodeAttempts[game_state] += 1
                #next have to implement giving the player a more/less difficult scenario based on the quality of their answer
                
            if str(input("Keep learning? (y/n)  ")).lower() == "n":
                print(f"final score: {difficulty_index} / {len(difficulties) - 1}")
                playing = False

    
class KuhnNode:
    def __init__(self, history, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.history = history
        
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
        normalizing_sum = np.sum(self.strategySum)
        if normalizing_sum > 0:
            return self.strategySum / normalizing_sum
        else:
            return np.repeat(1 / self.NUM_ACTIONS, self.NUM_ACTIONS)
    
    def get_move(self, strategy):
        return np.random.choice(self.moves, p=strategy)

if __name__ == '__main__':
    time1 = time.time()
    game = MCCFR()
    game.train(iterations=100000)
    print(f"run time: {abs(time1 - time.time())}")
    print()
    game.train_player()
    