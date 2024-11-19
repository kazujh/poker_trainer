import numpy as np

class RPS:
    def __init__(self, num_actions=3, iterations=100):
        self.NUM_ACTIONS = num_actions
        self.actions = [0, 1, 2]
        self.regrets = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])
        self.regretSum = np.zeros(self.NUM_ACTIONS)
        self.strategy = self.get_strategy(self.regretSum)
        self.strategySum = np.zeros(self.NUM_ACTIONS)

        self.oppregretSum = np.zeros(self.NUM_ACTIONS)
        self.oppstrategy = self.get_strategy(self.oppregretSum)
        self.oppstrategySum = np.zeros(self.NUM_ACTIONS)
        
        self.iterations = iterations

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

    def get_move(self, weights):
        return np.random.choice(self.actions, p=weights)
    
    
    def play(self):
        move = self.get_move(self.strategy)
        oppmove = self.get_move(self.oppstrategy)
        for num, i in enumerate(self.regretSum):
            self.regretSum[num] += self.regrets[num][oppmove] - self.regrets[move][oppmove]
        for num, i in enumerate(self.oppregretSum):
            self.oppregretSum[num] += self.regrets[num][move] - self.regrets[oppmove][move]
        self.strategy = self.get_strategy(self.regretSum)
        self.oppstrategy = self.get_strategy(self.oppregretSum)

        
    def train(self):
        for _ in range(self.iterations):
            self.play()
            self.strategySum += self.strategy
            self.oppstrategySum += self.oppstrategy
        
        print(f"Average strategy: {self.get_average_strategy(self.strategySum)}")
        print(f"Opponent average strategy: {self.get_average_strategy(self.oppstrategySum)}")
        

    def get_average_strategy(self, strategySum):
        total = sum(strategySum)
        if total > 0:
            average_strategy = strategySum / total
        else:
            average_strategy = np.full(self.NUM_ACTIONS, 1.0 / self.NUM_ACTIONS)
        return average_strategy

        
game = RPS(iterations=10000)
game.train()

"""import random
from collections import defaultdict

class SimplifiedTexasHoldem:
    def __init__(self, num_players=6):
        self.num_players = num_players
        self.deck = [r + s for r in '23456789TJQKA' for s in 'CDHS']
        self.players = [self.Player(i) for i in range(num_players)]
        self.reset()

    class Player:
        def __init__(self, player_id):
            self.id = player_id
            self.hole_cards = []
            self.is_active = True
            self.stack = 100  # Simplified stack size

    class Node:
        def __init__(self, info_set):
            self.info_set = info_set
            self.regret_sum = np.zeros(2)  # [Fold, Call/Raise]
            self.strategy = np.zeros(2)
            self.strategy_sum = np.zeros(2)

        def get_strategy(self, realization_weight):
            positive_regrets = np.maximum(self.regret_sum, 0)
            normalizing_sum = np.sum(positive_regrets)
            if normalizing_sum > 0:
                self.strategy = positive_regrets / normalizing_sum
            else:
                self.strategy = np.full(2, 0.5)
            self.strategy_sum += realization_weight * self.strategy
            return self.strategy

        def get_average_strategy(self):
            normalizing_sum = np.sum(self.strategy_sum)
            if normalizing_sum > 0:
                return self.strategy_sum / normalizing_sum
            else:
                return np.full(2, 0.5)

    def reset(self):
        self.pot = 0
        self.community_cards = []
        self.deck_copy = self.deck.copy()
        random.shuffle(self.deck_copy)
        for player in self.players:
            player.hole_cards = []
            player.is_active = True
        self.node_map = {}
        self.bet_size = 10  # Simplified fixed bet size

    def deal_hole_cards(self):
        for player in self.players:
            if player.is_active:
                player.hole_cards = [self.deck_copy.pop(), self.deck_copy.pop()]

    def deal_community_cards(self, num_cards):
        for _ in range(num_cards):
            self.community_cards.append(self.deck_copy.pop())

    def evaluate_hand(self, player):
        # Simplified hand evaluation: count high cards and pairs
        ranks = '23456789TJQKA'
        rank_values = {rank: i for i, rank in enumerate(ranks, 2)}
        hole_ranks = [card[0] for card in player.hole_cards]
        community_ranks = [card[0] for card in self.community_cards]
        all_ranks = hole_ranks + community_ranks
        counts = {rank: all_ranks.count(rank) for rank in set(all_ranks)}
        max_count = max(counts.values())
        if max_count >= 2:
            hand_strength = 10  # Pair or better
        else:
            # High card strength
            highest_rank = max([rank_values[rank] for rank in hole_ranks])
            hand_strength = highest_rank
        return hand_strength

    def is_terminal(self):
        active_players = [p for p in self.players if p.is_active]
        return len(active_players) <= 1 or len(self.community_cards) == 5

    def get_winner(self):
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1:
            return active_players[0]
        else:
            # Showdown
            best_hand = -1
            winner = None
            for player in active_players:
                hand_strength = self.evaluate_hand(player)
                if hand_strength > best_hand:
                    best_hand = hand_strength
                    winner = player
            return winner

    def get_info_set(self, player):
        hole_cards = ''.join(sorted([card[0] for card in player.hole_cards]))
        community = ''.join(sorted([card[0] for card in self.community_cards]))
        return f"P{player.id}|H{hole_cards}|C{community}"

    def cfr(self, player, history, probability):
        if self.is_terminal():
            winner = self.get_winner()
            if winner == player:
                return self.pot  # Winner takes the pot
            else:
                return -self.bet_size  # Simplified loss

        if not player.is_active:
            # If the player has folded, no need to proceed further
            return 0  # Utility is zero since the player cannot gain or lose more

        info_set = self.get_info_set(player)
        if info_set not in self.node_map:
            self.node_map[info_set] = self.Node(info_set)
        node = self.node_map[info_set]

        strategy = node.get_strategy(probability)
        util = np.zeros(2)
        node_util = 0

        actions = ['fold', 'call']
        for idx, action in enumerate(actions):
            # Save game state
            pot_backup = self.pot
            active_backup = [p.is_active for p in self.players]
            community_backup = self.community_cards.copy()
            stack_backup = player.stack

            # Apply action
            if action == 'fold':
                player.is_active = False
            else:
                self.pot += self.bet_size
                player.stack -= self.bet_size

            # Recursion
            if player.is_active:
                # Deal next community card if applicable
                if len(self.community_cards) < 5:
                    self.deal_community_cards(1)
                util[idx] = -self.cfr(player, history + [action], probability * strategy[idx])
            else:
                # Player has folded; utility is zero
                util[idx] = 0

            # Restore game state
            self.pot = pot_backup
            player.stack = stack_backup
            self.community_cards = community_backup
            for p, active in zip(self.players, active_backup):
                p.is_active = active

            node_util += strategy[idx] * util[idx]

        # Update regrets
        for idx, action in enumerate(actions):
            regret = util[idx] - node_util
            node.regret_sum[idx] += probability * regret

        return node_util


    def train(self, iterations):
        util = 0
        for i in range(iterations):
            self.reset()
            self.deal_hole_cards()
            # For simplicity, we'll only train one player and assume others play randomly
            player = self.players[0]
            util += self.cfr(player, history=[], probability=1)
        print(f"Average game value for Player 0: {util / iterations}")

    def get_strategy(self):
        strategies = {}
        for info_set, node in self.node_map.items():
            avg_strategy = node.get_average_strategy()
            strategies[info_set] = avg_strategy
        return strategies

# Instantiate and train the agent
game = SimplifiedTexasHoldem(num_players=6)
game.train(iterations=1000)

# Get the computed strategies
strategies = game.get_strategy()

# Display some strategies
print("\nSample strategies for Player 0:")
for info_set, strategy in list(strategies.items())[:10]:
    print(f"Info set: {info_set}, Strategy: Fold {strategy[0]:.2f}, Call {strategy[1]:.2f}")
"""
