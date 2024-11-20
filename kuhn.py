import numpy as np

class Kuhn:
    cards = ["J", "Q", "K"]
    def __init__(self):
        self.playercard = "K"
        self.oppcards = self.cards.copy()
        self.oppcard = "J"
        self.action_dict = " "
        self.player = Kuhn_Node(self.playercard + self.action_dict, 0, self)
        self.opponent = Kuhn_Node(self.oppcard + self.action_dict, 1, self)
        self.player.game = self
        self.opponent.game = self
        self.actions = ["p", "b"]
    
    def deal(self):
        self.playercard = np.random.choice(self.cards)

        self.oppcards = self.cards.copy()
        self.oppcards.remove(self.playercard)
        self.oppcard = np.random.choice(self.oppcards) 

        self.player = Kuhn_Node(self.playercard + " pb", 0)
        self.opponent = Kuhn_Pass()   #test measure 
        
        self.player.game = self
        self.opponent.game = self
    
    def is_terminal(self, action_dict: str):
        if action_dict[-2:] == "pp" or action_dict[-2:] == "bb" or action_dict[-2:] == "bp":
            return True
        else:
            return False
    
    def cfr(self, action_dict: str, player):
        #1 run a single game tree
        #2 calculate regret for the current node by doing hypothetical rewards - actual rewards
        #3 pass the actual reward up the game tree to calculate regret for higher up nodes
        #4 save regrets in the regret_table
        #5 be able to run another game tree with the new regret_tables for each node, and updated strategies

        #should simulate a single game
        #if the node is terminal, then I return the node
        #have self.play only run a single time, since cfr is the one that is going to be initiating the recursion/multiple iterations
        #need to incorporate reach probability
        if self.is_terminal(self.action_dict):
            reward_tup = self.showdown(self.action_dict)
            print(f"Reward_tupe: {reward_tup}")
            return np.array(self.calculate_regret(self.action_dict, reward_tup))
        else:
            next_node = self.move(player, self.opponent)
            print(f"Current player.regret_table: {player.regret_table}")
            player.regret_table[self.playercard] += np.flip(self.cfr(self.action_dict, next_node))
            player.strategy = player.get_strategy(player.regret_table)
            return player.regret_table[self.playercard]

    def calculate_reward(self, reward_tup: tuple):
        reward = reward_tup[0] - reward_tup[1]
        return reward
    
    def calculate_regret(self, action_dict: str, reward_tup: tuple):
        actual_reward = self.calculate_reward(reward_tup)
        test_action, new_action_dict = action_dict[-1], action_dict[0:len(action_dict)-1]
        hypothetical_reward_tup = self.showdown(new_action_dict + self.actions[(self.actions.index(test_action) + 1) % 2])
        hypothetical_reward = self.calculate_reward(hypothetical_reward_tup)
        
        if action_dict[-1:] == "p":
            return [actual_reward, hypothetical_reward - actual_reward]
        else:
            return [hypothetical_reward - actual_reward, actual_reward]
        #need to change this to be more efficient and well-written
        #adds to the current regret table for the node
        #still have not implemented traversing up the game tree to set the regret tables for the other nodes
        #this works for one " pbp" but need to bring it up the game tree to find the regrets for other nodes
        #need to pass this into the node so that it sets the values of the node's regrets to it


    def showdown(self, history: str):
        if history[-2:] == "bp":
            if len(history) == 4:
                return (1, 0)
            return (0, 1)
        else:
            if history[-2:] =="bb":
                if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                    return (2, 0)
                else:
                    return (0, 2)
            else:
                if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                    return (1, 0)
                else:
                    return (0, 1)
    
    """def get_regret(self, player, opponent):
        if player.is_terminal():
            return self.showdown()
        else:
            player.regretSum[0] += self.play(player.bet_node, opponent)[0]
            player.regretSum[1] += self.play(player.pass_node, opponent)[1]
            player.strategy = player.get_strategy(player.regretSum)
            opponent.strategy = opponent.get_strategy(opponent.regretSum)
            return player.regretSum"""
    
    def move(self, player, opponent):
        #still need to code updating the strategies
        #need to somehow figure out communicating between nodes
        #need to not create a new node every time this is run, as to accumulate regrets and strategies
        #remove the self. in front of the opp_node and the player_node_2, I only used it to check values
        #regret = hypothetical rewards - actual reward
        #cfr function has to actually play and then calculate rewards retroactively

        #new code for making it only play a single node of the game, and have a way to access the node or the action_dict
        move = player.get_move(player.strategy)
        self.action_dict += move
        return player.children[move]
        #code below
        '''move = player.get_move(player.strategy)
        print(f"Player move: {move}")
        turn = self.action_dict + move
        print(f"Current actions: {turn}")
        self.opp_node = Kuhn_Node(self.oppcard + turn, 1, self.player)
        self.opp_node.game = self
        opp_move = self.opp_node.get_move(self.opp_node.strategy)
        print(f"Opponent move: {opp_move}")
        turn = turn + opp_move
        print(f"Current actions: {turn}")
        if self.is_terminal(turn):
            opp_regrets = self.opp_node.get_regrets()
            print("end of actions")
            print(f"""FINAL REGRET TABLE FOR OPPONENT:
                  {opp_regrets}""")
        else:
            self.player_node_2 = Kuhn_Node(self.playercard + turn, 0, self.opp_node)
            self.player_node_2.game = self
            move_2 = self.player_node_2.get_move(self.player_node_2.strategy)
            print(f"Player moves again: {move}")
            turn = turn + move_2
            print(f"Final actions: {turn}")
            if self.is_terminal(turn):
                player_regrets = self.player_node_2.get_regrets()
            print("end of actions")
            print(f"""FINAL REGRET TABLE FOR PLAYER:
                  {player_regrets}""")'''
        
        



#bet node or pass node
class Kuhn_Node:
    def __init__(self, action_dict, player, parent_node=None, num_actions=2):
        self.NUM_ACTIONS = num_actions 
        self.action_dict = action_dict
        self.parent_node = parent_node
        self.game = None
        self.card = action_dict[:1]
        self.player = player
        if not self.is_terminal():
            self.bet_node = Kuhn_Node(action_dict + "b", (self.player + 1) % 2, parent_node=self)
            self.pass_node = Kuhn_Node(action_dict + "p", (self.player + 1) % 2, parent_node=self)
            self.children = {"b": self.bet_node, "p": self.pass_node}
        if self.action_dict != "":
            self.card = self.action_dict[0]
        self.regret_table = {
            "J":np.zeros(self.NUM_ACTIONS),
            "Q":np.zeros(self.NUM_ACTIONS),
            "K":np.zeros(self.NUM_ACTIONS)
        }
        self.actions = ["p", "b"]
        self.strategy = self.get_strategy(self.regret_table)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.reach_prob = 0
        self.reach_probSum = 0
    
    def get_strategy(self, regret_sum):
        for i in regret_sum.keys():
            regret_sum[i][regret_sum[i] < 0] = 0
        strategy = regret_sum.copy()
        for i in regret_sum.keys():
            normalizing_sum = sum(regret_sum[i])
            for a in range(self.NUM_ACTIONS):
                if normalizing_sum > 0:
                    strategy[i][a] /= normalizing_sum
                else:
                    strategy[i][a] = 1.0 / self.NUM_ACTIONS
        return strategy
    
    def get_average_strategy(self):
        if sum(self.strategySum) > 0:
            average_strategy = self.strategySum / sum(self.strategySum)
        else:
            average_strategy = np.full(self.NUM_ACTIONS, 1.0 / self.NUM_ACTIONS)
        return average_strategy
    
    def get_move(self, weights):
        weights = weights[self.card]
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
    
    def is_terminal(self):
        if self.action_dict[-2:] == "pp" or self.action_dict[-2:] == "bb" or self.action_dict[-2:] == "bp":
            return True
        else:
            return False




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

            



