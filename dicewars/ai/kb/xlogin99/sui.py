import logging

from dicewars.ai.kb.move_selection import get_transfer_to_border
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.kb.xlogin99.max_n import SimulationAI


class SuiAI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.max_transfers = max_transfers
        self.players_order = players_order

    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        # Firstly transfer to border
        if nb_transfers_this_turn < self.max_transfers:
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        players = []
        for player in self.players_order:
            if board.get_player_areas(player):
                players.append(player)
        # max_n for attack move
        simulation_ai = SimulationAI(players, self.max_transfers - nb_transfers_this_turn)
        _, move = simulation_ai.max_n(board, 1, self.player_name)
        return move
