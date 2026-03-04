import random
from collections import defaultdict

class RockPaperScissors:
    def __init__(self, learning=False):
        self.choices = ['rock', 'paper', 'scissors']
        self.user_score = 0
        self.computer_score = 0
        self.learning = learning

        # For the learning AI - tracks user patterns
        self.user_history = []
        self.pattern_counts = defaultdict(lambda: {'rock': 0, 'paper': 0, 'scissors': 0})

    def get_computer_choice(self):
        """Computer makes a choice - random for basic, or based on patterns if learning."""
        if not self.learning or len(self.user_history) < 2:
            # Random choice
            return random.choice(self.choices)

        # Learning mode: predict user's next move and counter it
        last_user_choice = self.user_history[-1]

        # Get statistics on what user plays after their previous moves
        if last_user_choice in self.pattern_counts:
            stats = self.pattern_counts[last_user_choice]
            total = sum(stats.values())

            if total > 0:
                # Find the most likely next choice by user
                likely_choice = max(stats.items(), key=lambda x: x[1])[0]

                # Counter the likely choice
                counters = {
                    'rock': 'paper',
                    'paper': 'scissors',
                    'scissors': 'rock'
                }
                return counters[likely_choice]

        # Fallback to random if not enough data
        return random.choice(self.choices)

    def determine_winner(self, user_choice, computer_choice):
        """Determine the winner of a single round."""
        if user_choice == computer_choice:
            return 'tie'

        # Winning conditions
        if (user_choice == 'rock' and computer_choice == 'scissors' or
            user_choice == 'paper' and computer_choice == 'rock' or
            user_choice == 'scissors' and computer_choice == 'paper'):
            return 'user'
        else:
            return 'computer'

    def play_round(self, user_choice):
        """Play a single round."""
        user_choice = user_choice.lower()

        if user_choice not in self.choices:
            return None, "Invalid choice! Please enter 'rock', 'paper', or 'scissors'."

        computer_choice = self.get_computer_choice()
        result = self.determine_winner(user_choice, computer_choice)

        # Update learning data
        if self.learning and len(self.user_history) > 0:
            self.pattern_counts[self.user_history[-1]][user_choice] += 1
        self.user_history.append(user_choice)

        # Update scores
        if result == 'user':
            self.user_score += 1
            message = f"You win! You chose {user_choice}, Computer chose {computer_choice}."
        elif result == 'computer':
            self.computer_score += 1
            message = f"Computer wins! You chose {user_choice}, Computer chose {computer_choice}."
        else:
            message = f"It's a tie! You both chose {user_choice}."

        return result, message

    def display_scores(self):
        """Display current scores."""
        print(f"\n--- Score ---")
        print(f"You: {self.user_score}")
        print(f"Computer: {self.computer_score}")
        print(f"------------ \n")

    def get_mode_info(self):
        """Display game mode information."""
        if self.learning:
            return "LEARNING MODE: The AI learns from your patterns!"
        else:
            return "BASIC MODE: The AI makes random choices."

def play_game(learning=False):
    """Main game loop."""
    game = RockPaperScissors(learning=learning)

    print("\n" + "="*50)
    print("ROCK-PAPER-SCISSORS GAME")
    print("="*50)
    print(game.get_mode_info())
    print("Type 'rock', 'paper', or 'scissors' to play")
    print("Type 'quit' to exit, 'score' to see scores")
    print("="*50 + "\n")

    while True:
        user_input = input("Your choice: ").strip()

        if user_input.lower() == 'quit':
            print("\nThanks for playing!")
            game.display_scores()
            break
        elif user_input.lower() == 'score':
            game.display_scores()
            continue

        result, message = game.play_round(user_input)

        if result is not None:
            print(message)
            game.display_scores()
        else:
            print(message)

if __name__ == "__main__":
    print("\nChoose game mode:")
    print("1. Basic (Random AI)")
    print("2. Learning (AI learns your patterns)")

    mode = input("Enter 1 or 2: ").strip()

    if mode == '2':
        play_game(learning=True)
    else:
        play_game(learning=False)
