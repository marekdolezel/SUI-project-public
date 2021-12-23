import logging

from dicewars.ai.kb.move_selection import get_transfer_to_border, get_transfer_from_endangered
from dicewars.ai.utils import possible_attacks, probability_of_successful_attack, probability_of_holding_area
from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, TransferCommand
from dicewars.ai.kb.xlogin99.max_n import DepthFirstSearch
import numpy as np
from matplotlib import pyplot as plt
from dicewars.ai.kb.xlogin99.gameSerialize import *

class SuiAI:
    def __init__(self, player_name, board, players_order, max_transfers):
        self.player_name = player_name
        self.max_transfers = max_transfers
        self.players_order = players_order
        self.write_to_file = False # Change this to True if you want to write gameStateVector to a file

    """ Improved STEi_ADT agent: 
    1. Firstly perform aggressive transfer (STEi_ADT).
    2. Perform Depth First Search with max-N algorithm to choose the best attack.
    3. At last, perform defensive transfer (STEi_ADT).
    """
    def ai_turn(self, board, nb_moves_this_turn, nb_transfers_this_turn, nb_turns_this_game, time_left):
        # Serialize the game and save the array to fileName.np
        if self.write_to_file == True:
            M, o, d = serialize_game_state(board)
            serializedGame = game_state_to_vector(M, o, d)
            save_game_state_vector_to_file(serializedGame)
            self.write_to_file = False

        # Firstly perform 4 transfer moves into the border
        if nb_transfers_this_turn < self.max_transfers - 2:
            transfer = get_transfer_to_border(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])
        players = []
        for player in self.players_order:
            if board.get_player_areas(player):
                players.append(player)

        # search max_n to choose the best attack move
        move_depth = 1  # each player perform multiple attack moves => depth of tested moves
        player_depth = 3  # depth 3 => 1. turn of 4 players, depth 6 => 2. turns of 3 players
        dft = DepthFirstSearch(players)
        moves = dft.max_n(board, move_depth, player_depth, self.player_name)
        if len(moves) != 0:
            return moves[0]

        # At last, perform 2 transfer moves out of the border
        if nb_transfers_this_turn < self.max_transfers:
            transfer = get_transfer_from_endangered(board, self.player_name)
            if transfer:
                return TransferCommand(transfer[0], transfer[1])

        return EndTurnCommand()
