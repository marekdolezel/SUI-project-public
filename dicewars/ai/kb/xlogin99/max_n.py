import copy

from dicewars.ai.kb.move_selection import get_transfer_to_border
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand


class SimulationAI:
    def __init__(self, players_order, max_transfers):
        self.player_order = players_order
        self.max_transfers = max_transfers

    """ Max-N algorithm from
    https://docplayer.net/131108557-Expectimax-n-a-modification-of-expectimax-algorithm-to-solve-multiplayer-stochastic-game.html
    """
    def max_n(self, board, depth, player):
        if depth == 0:
            return self.evaluate(board), None
        score = [float('-inf'), float('-inf'), float('-inf'), float('-inf')]
        best_move = EndTurnCommand()
        for move in self.possible_moves(board, player):
            board_copy = copy.deepcopy(board)
            board = self.simulate_ai(board, move)
            value, _ = self.max_n(board, depth-1, self.next_player())
            board = board_copy  # undo simulation
            if value[player] > score[player]:
                score = value
                best_move = move
        return score, best_move

    def next_player(self):
        now = self.player_order.pop(0)
        self.player_order.append(now)
        return self.player_order[0]

    def possible_moves(self, board, player):
        turns = []
        for source, target in possible_attacks(board, player):
            area_name = source.get_name()
            atk_power = source.get_dice()
            atk_prob = probability_of_successful_attack(board, area_name, target.get_name())
            hold_prob = atk_prob * probability_of_holding_area(board, target.get_name(), atk_power - 1, player)
            if hold_prob >= 0.4 or atk_power == 8:
                turns.append([area_name, target.get_name(), hold_prob])
        attacks = sorted(turns, key=lambda turn: turn[2], reverse=True)
        attacks = [BattleCommand(attack[0], attack[1]) for attack in attacks]
        return attacks

    """ Apply move to board - Firstly transfers, then attacks.
    """
    def simulate_ai(self, board, action):
        #for action in move:
            # if isinstance(action, TransferCommand):
            #     transfer = action
            #     src_dice = transfer.source_name.get_dice()
            #     dst_dice = transfer.target_name.get_dice()
            #     dice_moved = min(8 - dst_dice, src_dice - 1)
            #     board.get_area(transfer.source_name).set_dice(src_dice - dice_moved)
            #     board.get_area(transfer.target_name).set_dice(dst_dice + dice_moved)
        attack = action
        attacker = board.get_area(attack.source_name)
        defender = board.get_area(attack.target_name)
        atk_dice = attacker.get_dice()
        def_dice = defender.get_dice()
        atk_pwr = def_pwr = 0
        atk_name = attacker.get_owner_name()
        for i in range(0, atk_dice):
            atk_pwr += 3
        for i in range(0, def_dice):
            def_pwr += 3
        attacker.set_dice(1)
        if atk_pwr > def_pwr:
            defender.set_owner(atk_name)
            attacker.set_dice(1)
            defender.set_dice(atk_dice - 1)
        else:
            battle_wear_min = 0  # TODO what is this value?
            battle_wear = atk_dice  # // battle_wear_min
            def_dice_left = max(1, def_dice - battle_wear)
            defender.set_dice(def_dice_left)
        return board

    """ Evaluate terminal node of maxN.
    Value of terminal node is computed as the largest region on the board for given player.
    """
    def evaluate(self, board):
        value = dict.fromkeys(range(1, 4))
        for player in self.player_order:
            players_regions = board.get_players_regions(player)
            max_region_size = max(len(region) for region in players_regions)
            value[player] = max_region_size
        return value

