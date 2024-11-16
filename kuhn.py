import numpy as np

class Kuhn:
    cards = ["J", "Q", "K"]
    def __init__(self):
        self.playercard = "J"
        self.oppcards = self.cards.copy()
        self.oppcard = "K"
        self.action_dict = " "
        self.player = Kuhn_Node(self.playercard + self.action_dict, 0, self)
        self.opponent = Kuhn_Node(self.oppcard + self.action_dict, 1, self)
        self.player.game = self
        self.opponent.game = self
    
    def deal(self):
        self.playercard = np.random.choice(self.cards)

        self.oppcards = self.cards.copy()
        self.oppcards.remove(self.playercard)
        self.oppcard = np.random.choice(self.oppcards) 

        self.player = Kuhn_Node(self.playercard + " pb", 0)
        self.opponent = Kuhn_Pass()   #test measure 
        
        self.player.game = self
        self.opponent.game = self
    
    def is_terminal(self, action_dict):
        if action_dict[-2:] == "pp" or action_dict[-2:] == "bb" or action_dict[-2:] == "bp":
            return True
        else:
            return False
    
    def cfr(self, history):
        #1 run a single game tree
        #2 calculate regret for the current node by doing hypothetical rewards - actual rewards
        #3 pass the actual reward up the game tree to calculate regret for higher up nodes
        #4 save regrets in the regret_table
        #5 be able to run another game tree with the new regret_tables for each node, and updated strategies

        #should simulate a single game
        #if the node is terminal, then I return the node
        #have self.play only run a single time, since cfr is the one that is going to be initiating the recursion/multiple iterations
        self.play(self.player, self.opponent)

        #access the final node
    
    def showdown(self, history):
        if history[-2:] == "bp":
            if len(history) == 4:
                return (2, 0)
            return (0, 1)
        else:
            if self.cards.index(self.playercard) > self.cards.index(self.oppcard):
                return (2, 0)
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
    
    def play(self, player, opponent):
        #still need to code updating the strategies
        #need to somehow figure out communicating between nodes
        #need to not create a new node every time this is run, as to accumulate regrets and strategies
        #remove the self. in front of the opp_node and the player_node_2, I only used it to check values
        #regret = hypothetical rewards - actual reward
        #cfr function has to actually play and then calculate rewards retroactively

        #new code
        
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
        if self.action_dict != "":
            self.card = self.action_dict[0]
        self.regret_table = {
            "J":{"b": 0.0, "p": 0.0},
            "Q":{"b": 0.0, "p": 0.0},
            "K":{"b": 0.0, "p": 0.0}
        }
        self.actions = ["p", "b"]
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategy = self.get_strategy(self.regretSum)
        self.strategySum = np.zeros(self.NUM_ACTIONS)
        self.reach_prob = 0
        self.reach_probSum = 0
    
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

            



