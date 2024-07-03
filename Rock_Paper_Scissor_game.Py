%%writefile RPS.py
def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    # Strategy Initialization
    if len(opponent_history) < 5:
        # Start with a random move if there's not enough data
        return "R"
    
    # Analyze opponent's history
    move_counts = {"R": 0, "P": 0, "S": 0}
    for move in opponent_history:
        move_counts[move] += 1
    
    most_frequent_move = max(move_counts, key=move_counts.get)
    
    # Counter the most frequent move
    if most_frequent_move == "R":
        return "P"
    elif most_frequent_move == "P":
        return "S"
    else:
        return "R"

# Optional: Reset the opponent history for fresh matches
def reset_player():
    global opponent_history
    opponent_history = []
%%writefile RPS_game.py
import random

def play(player1, player2, num_games, verbose=False):
    p1_history = []
    p2_history = []
    p1_wins = 0
    p2_wins = 0
    ties = 0

    for _ in range(num_games):
        p1_move = player1(p2_history[-1] if p2_history else "")
        p2_move = player2(p1_history[-1] if p1_history else "")
        
        p1_history.append(p1_move)
        p2_history.append(p2_move)
        
        if p1_move == p2_move:
            ties += 1
            result = "Tie"
        elif (p1_move == "R" and p2_move == "S") or (p1_move == "P" and p2_move == "R") or (p1_move == "S" and p2_move == "P"):
            p1_wins += 1
            result = "Player 1 wins"
        else:
            p2_wins += 1
            result = "Player 2 wins"
        
        if verbose:
            print(f"Player 1: {p1_move}  Player 2: {p2_move}  ->  {result}")

    print(f"Final result: Player 1: {p1_wins}, Player 2: {p2_wins}, Ties: {ties}")

def quincy(prev_play):
    return "R"

def abbey(prev_play, moves=["P", "P", "S", "R", "R"]):
    if prev_play:
        abbey.history.append(prev_play)
    else:
        abbey.history = []
    return moves[len(abbey.history) % len(moves)]

def kris(prev_play):
    if not hasattr(kris, "counter"):
        kris.counter = 0
    kris.counter += 1
    return ["R", "P", "S"][kris.counter % 3]

def mrugesh(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    if len(opponent_history) < 2:
        return "R"

    last_two = opponent_history[-2:]
    if last_two == ["R", "R"]:
        return "P"
    elif last_two == ["P", "P"]:
        return "S"
    elif last_two == ["S", "S"]:
        return "R"
    elif last_two == ["R", "P"]:
        return "S"
    elif last_two == ["R", "S"]:
        return "P"
    elif last_two == ["P", "R"]:
        return "S"
    elif last_two == ["P", "S"]:
        return "R"
    elif last_two == ["S", "R"]:
        return "P"
    elif last_two == ["S", "P"]:
        return "R"
    return random.choice(["R", "P", "S"])
%%writefile main.py
from RPS_game import play, quincy, abbey, kris, mrugesh
from RPS import player

# Example of playing the game with logging enabled
print("Playing against Quincy")
play(player, quincy, 1000, verbose=True)
print("\nPlaying against Abbey")
play(player, abbey, 1000, verbose=True)
print("\nPlaying against Kris")
play(player, kris, 1000, verbose=True)
print("\nPlaying against Mrugesh")
play(player, mrugesh, 1000, verbose=True)

# Uncomment the following line to run unit tests automatically
# import test_module
%%writefile test_module.py
import unittest
from RPS import player

class TestPlayer(unittest.TestCase):
    def test_initial_move(self):
        self.assertIn(player(""), ["R", "P", "S"], "The initial move should be 'R', 'P', or 'S'.")

    def test_following_moves(self):
        history = ["R", "P", "S", "R", "P", "S"]
        for move in history:
            self.assertIn(player(move), ["R", "P", "S"], "The following moves should be 'R', 'P', or 'S'.")

if __name__ == "__main__":
    unittest.main()


!python main.py


!python test_module.py
play(player, quincy, 1000, verbose=True)
