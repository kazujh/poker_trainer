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

        
game = RPS(iterations=1000)
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
import random
import numpy as np

class Player:
    def __init__(self, player_id):
        self.id = player_id
        self.hole_cards = []
        self.is_active = True
        self.stack = 100  # Simplified stack size
        self.has_acted = False

    def clone(self):
        cloned_player = Player(self.id)
        cloned_player.hole_cards = self.hole_cards.copy()
        cloned_player.is_active = self.is_active
        cloned_player.stack = self.stack
        cloned_player.has_acted = self.has_acted
        return cloned_player

class GameState:
    def __init__(self, players, pot, community_cards, deck, current_player, bet_size):
        self.players = players  # List of Player objects
        self.pot = pot
        self.community_cards = community_cards
        self.deck = deck
        self.current_player = current_player  # Player ID
        self.bet_size = bet_size

    def clone(self):
        # Create a deep copy of the game state
        cloned_players = [player.clone() for player in self.players]
        cloned_pot = self.pot
        cloned_community_cards = self.community_cards.copy()
        cloned_deck = self.deck.copy()
        return GameState(cloned_players, cloned_pot, cloned_community_cards, cloned_deck, self.current_player, self.bet_size)

    def is_terminal(self):
        # Game is terminal if only one active player remains or all community cards are dealt and betting is over
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) <= 1:
            return True
        elif len(self.community_cards) == 5 and all(not p.is_active or p.has_acted for p in self.players):
            return True
        else:
            return False

    def get_utility(self, player_id):
        player = self.players[player_id]
        if not player.is_active:
            return - (100 - player.stack)  # Net loss of the amount bet
        winners = self.get_winner()
        if player in winners:
            return self.pot / len(winners) - (100 - player.stack)  # Net gain
        else:
            return - (100 - player.stack)  # Net loss

    def get_winner(self):
        # Determine the winner(s)
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1:
            return active_players
        else:
            # Showdown
            best_hand = -1
            winners = []
            for player in active_players:
                hand_strength = self.evaluate_hand(player)
                if hand_strength > best_hand:
                    best_hand = hand_strength
                    winners = [player]
                elif hand_strength == best_hand:
                    winners.append(player)
            return winners

    def apply_action(self, action):
        # Apply the action to the game state
        player = self.players[self.current_player]
        if action == 'fold':
            player.is_active = False
        elif action == 'call':
            player.stack -= self.bet_size
            self.pot += self.bet_size
        else:
            raise ValueError("Invalid action")

        # Mark that the player has acted
        player.has_acted = True

        # Check if the betting round is over
        if all(not p.is_active or p.has_acted for p in self.players):
            if self.is_terminal():
                # Game is over; no need to reset has_acted or proceed further
                return
            else:
                # Proceed to the next game phase
                self.advance_game_phase()
                # Reset has_acted flags for the next betting round
                for p in self.players:
                    p.has_acted = False
                # Reset current_player to the first active player
                self.current_player = self.find_first_active_player()
        else:
            # Move to the next player
            self.move_to_next_player()

    def move_to_next_player(self):
        starting_player = self.current_player
        while True:
            self.current_player = (self.current_player + 1) % len(self.players)
            if self.players[self.current_player].is_active and not self.players[self.current_player].has_acted:
                break
            if self.current_player == starting_player:
                break  # All players have acted

    def advance_game_phase(self):
        if len(self.community_cards) < 5:
            # Deal one more community card
            self.community_cards.append(self.deck.pop())
        # No need to reset current_player here; it's handled in apply_action

    def find_first_active_player(self):
        for i, player in enumerate(self.players):
            if player.is_active:
                return i
        return 0

    def get_info_set(self, player_id):
        player = self.players[player_id]
        hole_cards = ''.join(sorted([card[0] for card in player.hole_cards]))
        community = ''.join(sorted([card[0] for card in self.community_cards]))
        return f"P{player_id}|H{hole_cards}|C{community}|Actions{self.get_betting_history()}"

    def get_betting_history(self):
        # Simplified betting history
        return ''

    def evaluate_hand(self, player):
        # Simplified hand evaluation
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
            highest_rank = max([rank_values[rank] for rank in hole_ranks])
            hand_strength = highest_rank
        return hand_strength

class SimplifiedTexasHoldem:
    def __init__(self, num_players=6):
        self.num_players = num_players
        self.deck_template = [r + s for r in '23456789TJQKA' for s in 'CDHS']
        self.players = [Player(i) for i in range(num_players)]
        self.node_map = [{} for _ in range(num_players)]
        self.bet_size = 10  # Simplified fixed bet size

    class Node:
        def __init__(self, info_set):
            self.info_set = info_set
            self.regret_sum = np.zeros(2)  # [Fold, Call]
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
        self.deck = self.deck_template.copy()
        random.shuffle(self.deck)
        for player in self.players:
            player.hole_cards = []
            player.is_active = True
            player.stack = 100  # Reset stack
            player.has_acted = False

    def deal_hole_cards(self):
        for player in self.players:
            player.hole_cards = [self.deck.pop(), self.deck.pop()]

    def deal_community_cards(self, num_cards):
        for _ in range(num_cards):
            self.community_cards.append(self.deck.pop())

    def train(self, iterations):
        for _ in range(iterations):
            self.reset()
            self.deal_hole_cards()
            self.deal_community_cards(3)  # Deal flop
            initial_state = GameState(
                players=self.players,
                pot=self.pot,
                community_cards=self.community_cards.copy(),
                deck=self.deck.copy(),
                current_player=0,
                bet_size=self.bet_size
            )
            reach_probs = [1.0] * self.num_players
            self.cfr(initial_state, reach_probs)

    def cfr(self, game_state, reach_probs):
        if game_state.is_terminal():
            utilities = [game_state.get_utility(p.id) for p in game_state.players]
            return utilities

        current_player_id = game_state.current_player
        player = game_state.players[current_player_id]
        info_set = game_state.get_info_set(current_player_id)

        if info_set not in self.node_map[current_player_id]:
            self.node_map[current_player_id][info_set] = self.Node(info_set)
        node = self.node_map[current_player_id][info_set]

        strategy = node.get_strategy(reach_probs[current_player_id])
        util = np.zeros(len(strategy))
        node_util = 0

        actions = ['fold', 'call']
        for idx, action in enumerate(actions):
            next_game_state = game_state.clone()
            next_reach_probs = reach_probs.copy()
            next_reach_probs[current_player_id] *= strategy[idx]

            # Apply action in the cloned game state
            next_game_state.apply_action(action)

            # Recursive call
            util_vector = self.cfr(next_game_state, next_reach_probs)

            if util_vector is None or len(util_vector) != self.num_players:
                raise ValueError(f"Invalid util_vector returned from cfr: {util_vector}")

            util[idx] = util_vector[current_player_id]
            node_util += strategy[idx] * util[idx]

        # Update regrets
        opponent_reach = np.prod([reach_probs[i] for i in range(self.num_players) if i != current_player_id])
        for idx, action in enumerate(actions):
            regret = util[idx] - node_util
            node.regret_sum[idx] += opponent_reach * regret

        return_values = [0] * self.num_players
        return_values[current_player_id] = node_util
        return return_values

    def get_strategy(self, player_id):
        strategies = {}
        for info_set, node in self.node_map[player_id].items():
            avg_strategy = node.get_average_strategy()
            strategies[info_set] = avg_strategy
        return strategies

# Instantiate and train the agent
game = SimplifiedTexasHoldem(num_players=6)
game.train(iterations=1000)

# Get the computed strategies
strategies = game.get_strategy(player_id=0)

# Display some strategies
print("\nSample strategies for Player 0:")
for info_set, strategy in list(strategies.items())[:10]:
    print(f"Info set: {info_set}, Strategy: Fold {strategy[0]:.2f}, Call {strategy[1]:.2f}")
