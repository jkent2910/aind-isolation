"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

import random
from copy import deepcopy

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    their_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(my_moves - (2 * their_moves))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    my_distance_from_center = float((h - y)**2 + (w - x)**2)

    a, b = game.get_player_location(game.get_opponent(player))
    their_distance_from_center = float((h - a)**2 + (w - b)**2)

    return float(their_distance_from_center - my_distance_from_center)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = game.get_legal_moves(player)
    my_wall_moves = [m for m in my_moves if m[0] == 0 or m[0] == (game.height - 1) or m[1] == 0 or m[1] == (game.width - 1)]

    their_moves = game.get_legal_moves(game.get_opponent(player))
    their_wall_moves = [m for m in their_moves if m[0] == 0 or m[0] == (game.height - 1) or m[1] == 0 or m[1] == (game.width - 1)]

    return float(len(their_wall_moves) - len(my_wall_moves))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        best_move = (-1, -1)
        best_score = float("-inf")

        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 1:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        best_score = float("inf")
        for move in legal_moves:
            score = self.max_value(game.forecast_move(move), depth-1)
            if score < best_score:
                best_score = score

        return best_score

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 1:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        best_score = float("-inf")
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth - 1)
            if score > best_score:
                best_score = score

        return best_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for depth in range(1,100):
                best_move = self.alphabeta(game, depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()

        if legal_moves:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1)
        best_score = float("-inf")

        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if best_score >= beta:
                return best_move
        return best_move

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 1:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        best_score = float("inf")
        for move in legal_moves:
            score = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)
            if score < best_score:
                best_score = score

            # Prune?
            if best_score <= alpha:
                return best_score

            # Update beta?
            beta = min(beta, best_score)
        return best_score

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth == 1:
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        best_score = float("-inf")
        for move in legal_moves:
            score = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if score > best_score:
                best_score = score

            # Prune?
            if best_score >= beta:
                return best_score

            # Update alpha?
            alpha = max(alpha, best_score)
        return best_score


            # if __name__ == "__main__":
#     from isolation import Board
#
#     # create an isolation board (by default 7x7)
#     player1 = MinimaxPlayer()
#     player2 = MinimaxPlayer()
#     game = Board(player1, player2)
#
#     # place player 1 on the board at row 2, column 3, then place player 2 on
#     # the board at row 0, column 5; display the resulting board state.  Note
#     # that the .apply_move() method changes the calling object in-place.
#     game.apply_move((2, 3))
#     game.apply_move((0, 5))
#     print(game.to_string())
#
#     # players take turns moving on the board, so player1 should be next to move
#     assert(player1 == game.active_player)
#
#     # get a list of the legal moves available to the active player
#     print(game.get_legal_moves())
#
#     # get a successor of the current state by making a copy of the board and
#     # applying a move. Notice that this does NOT change the calling object
#     # (unlike .apply_move()).
#     new_game = game.forecast_move((1, 1))
#     assert(new_game.to_string() != game.to_string())
#     print("\nOld state:\n{}".format(game.to_string()))
#     print("\nNew state:\n{}".format(new_game.to_string()))
#
#     # play the remainder of the game automatically -- outcome can be "illegal
#     # move", "timeout", or "forfeit"
#     winner, history, outcome = game.play()
#     print("\nWinner: {}\nOutcome: {}".format(winner, outcome))
#     print(game.to_string())
#     print("Move history:\n{!s}".format(history))
