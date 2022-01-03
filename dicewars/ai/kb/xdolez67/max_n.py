import copy

from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.client.ai_driver import BattleCommand

#AI
import torch
from torch import nn
# from dicewars.ai.kb.xdolez67.network import *
from dicewars.ai.kb.xdolez67.gameSerialize import serialize_game_stateNoTraslation
import sys,os


class NetworkSui(nn.Module):

    def init(self):
        super(NetworkSui, self).init()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(663, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Softmax(dim=1)
        )


class DepthFirstSearch:
    def __init__(self, players_order):
        self.player_order = players_order
        self.max_move_depth = 1
        self.ai_model = NetworkSui()
        self.model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.ai_model.load_state_dict(torch.load(self.model_path), strict=False)

    def max_n(self, board, move_depth, player_depth, player):
        self.max_move_depth = move_depth
        _, moves = self.max_n_recursion(board, move_depth, player_depth, player)
        return moves

    """ Max-N algorithm from
    https://docplayer.net/131108557-Expectimax-n-a-modification-of-expectimax-algorithm-to-solve-multiplayer-stochastic-game.html
    """
    def max_n_recursion(self, board, move_depth, player_depth, player):
        if player_depth == 0:
            return self.evaluate(board), []

        moves = self.possible_moves(board, player)
        if move_depth == 0 or len(moves) == 0:
            next_player = self.next_player()
            score, _ = self.max_n_recursion(board, self.max_move_depth, player_depth - 1, next_player)
            return score, []

        # find best move for player
        score = {1: float('-inf'), 2: float('-inf'), 3: float('-inf'), 4: float('-inf')}
        best_moves = []
        for move in moves:
            board_copy = copy.deepcopy(board)
            new_board = self.apply_move_to_board(board_copy, move)
            value, prev_moves = self.max_n_recursion(new_board, move_depth - 1, player_depth, player)
            if value[player] > score[player]:
                score = value
                best_moves = [move] + prev_moves

        return score, best_moves

    def next_player(self):
        now = self.player_order.pop(0)
        self.player_order.append(now)
        return self.player_order[0]

    """ Compute possible attack similarly to the stei AI.
    """
    def possible_moves(self, board, player):
        turns = []
        for source, target in possible_attacks(board, player):
            area_name = source.get_name()
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(board, area_name, target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(board, target.get_name(), atk_power - 1, player)
            if hold_prob >= 0.5 or atk_power == 8:
                turns.append([area_name, target.get_name(), hold_prob])
        attacks = sorted(turns, key=lambda turn: turn[2], reverse=True)
        attacks = [BattleCommand(attack[0], attack[1]) for attack in attacks]
        return attacks

    """ Apply move to board -> perform attack. Function is deterministic.
    If attacker posses more dices than defender then he win.
    """
    def apply_move_to_board(self, board, action):
        attack = action
        attacker = board.get_area(attack.source_name)
        defender = board.get_area(attack.target_name)
        atk_dice = attacker.get_dice()
        def_dice = defender.get_dice()
        atk_name = attacker.get_owner_name()
        attacker.set_dice(1)
        if atk_dice > def_dice:
            defender.set_owner(atk_name)
            attacker.set_dice(1)
            defender.set_dice(atk_dice - 1)
        else:
            battle_wear = atk_dice  # // battle_wear_min
            def_dice_left = max(1, def_dice - battle_wear)
            defender.set_dice(def_dice_left)
        return board

    """ Evaluate terminal node of maxN.
    Score of each player in the terminal node is computed as the largest region
    on the board for given player.
    """
    def evaluate(self, board):
        value = {1: float('-inf'), 2: float('-inf'), 3: float('-inf'), 4: float('-inf')}
        for player in self.player_order:
            players_regions = board.get_players_regions(player)
            if players_regions == 0:
                value[player] = 0
            else:
                value[player] = max(len(region) for region in players_regions)
        return value

    def evaluate_ai(self, board):
        gameVector = serialize_game_stateNoTraslation(board)
        self.ai_model.eval()
        value = self.ai_model(gameVector)
        return value
