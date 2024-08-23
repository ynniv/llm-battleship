import curses
import random
import sys
import requests
import json
import datetime
import re
import argparse
import os
from openai import OpenAI
from anthropic import Anthropic
import time

BOARD_SIZE = 8
SHIPS = [
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2)
]
MAX_GUESSES = BOARD_SIZE * BOARD_SIZE

def create_board():
    return [['O' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

def place_ship(board, ship):
    ship_name, ship_size = ship
    while True:
        is_horizontal = random.choice([True, False])
        if is_horizontal:
            row = random.randint(0, BOARD_SIZE - 1)
            col = random.randint(0, BOARD_SIZE - ship_size)
            if all(board[row][col+i] == 'O' for i in range(ship_size)):
                for i in range(ship_size):
                    board[row][col+i] = 'S'
                return
        else:
            row = random.randint(0, BOARD_SIZE - ship_size)
            col = random.randint(0, BOARD_SIZE - 1)
            if all(board[row+i][col] == 'O' for i in range(ship_size)):
                for i in range(ship_size):
                    board[row+i][col] = 'S'
                return

def initialize_game():
    board = create_board()
    ships_positions = {}
    for ship, size in SHIPS:
        placed = False
        while not placed:
            is_horizontal = random.choice([True, False])
            if is_horizontal:
                row = random.randint(0, BOARD_SIZE - 1)
                col = random.randint(0, BOARD_SIZE - size)
                if all(board[row][col+i] == 'O' for i in range(size)):
                    for i in range(size):
                        board[row][col+i] = 'S'
                    ships_positions[ship] = [(row, col+i) for i in range(size)]
                    placed = True
            else:
                row = random.randint(0, BOARD_SIZE - size)
                col = random.randint(0, BOARD_SIZE - 1)
                if all(board[row+i][col] == 'O' for i in range(size)):
                    for i in range(size):
                        board[row+i][col] = 'S'
                    ships_positions[ship] = [(row+i, col) for i in range(size)]
                    placed = True
    return board, [['O' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)], ships_positions


def draw_board(win, board, hide_ships=True):
    win.clear()
    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLUE)  # Water
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_RED)    # Hit
    curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_WHITE)  # Miss
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_GREEN)  # Ship

    # Draw column numbers
    for i in range(BOARD_SIZE):
        win.addstr(0, i*4 + 5, str(i))

    for i, row in enumerate(board):
        # Draw row letters
        win.addstr(i*3 + 2, 0, chr(65 + i))
        for j, cell in enumerate(row):
            color = curses.color_pair(1)  # Default to water color
            if cell == 'M':
                color = curses.color_pair(3)
            elif cell == 'X':
                color = curses.color_pair(2)
            elif cell == 'S' and not hide_ships:
                color = curses.color_pair(4)

            win.addstr(i*3 + 1, j*4 + 3, "  ", color)
            win.addstr(i*3 + 2, j*4 + 3, "  ", color)

    win.refresh()

def get_clicked_cell(y, x):
    row = (y - 1) // 3
    col = (x - 3) // 4
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None

def extract_guess(response):
    pattern = r'\[GUESS\](.*?)\[/GUESS\]'
    matches = re.findall(pattern, response)
    if matches:
        return matches[0].strip()
    return None

def write_to_log(log_file, message):
    print(message, file=log_file, flush=True)
    return message

def create_board_string(board):
    header = "  " + " ".join(str(i) for i in range(1, BOARD_SIZE + 1))
    rows = [f"{chr(65+i)} {' '.join(row)}" for i, row in enumerate(board)]
    return header + "\n" + "\n".join(rows)

def process_guess(board, ai_board_state, row, col, ships_positions, log_file):
    guess_coord = f"{chr(row + ord('A'))}{col + 1}"
    result = ""
    if board[row][col] not in ['M', 'X']:
        if board[row][col] == 'S':
            board[row][col] = 'X'
            ai_board_state[row][col] = 'X'
            ships_sunk, ship_sunk = update_ships_positions(ships_positions, row, col)
            if ship_sunk:
                result = f"Hit and sunk {ship_sunk}!"
            else:
                result = "Hit!"
            write_to_log(log_file, f"Game: {result} at {guess_coord}")
        elif board[row][col] == 'O':
            board[row][col] = 'M'
            ai_board_state[row][col] = 'M'
            result = "Miss"
            write_to_log(log_file, f"Game: Miss at {guess_coord}.")
    else:
        result = "Already guessed"
        write_to_log(log_file, f"Game: Already guessed {guess_coord}, no change in game state.")
    return result

def update_ships_positions(ships_positions, row, col):
    for ship, positions in ships_positions.items():
        if (row, col) in positions:
            positions.remove((row, col))
            if not positions:
                return ship, True
            return ship, False
    return None, False

def ask_ollama(model, prompt):
    ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    response = requests.post(f"{ollama_url}/api/generate",
                             json={"model": model, "prompt": prompt},
                             stream=True)
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if 'response' in data:
                        full_response += data['response']
                except json.JSONDecodeError:
                    continue
        return full_response.strip()
    else:
        return None

def ask_openai(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

def ask_anthropic(client, model, prompt):
    try:
        if model.startswith("claude-3"):
            response = client.messages.create(
                model=model,
                max_tokens=1024,  # Increased token limit
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        else:
            response = client.completions.create(
                model=model,
                prompt=f"\n\nHuman: {prompt}\n\nAssistant:",
                max_tokens_to_sample=1024,  # Increased token limit
                temperature=0.5,
            )
            return response.completion
    except Exception as e:
        return f"Error: {str(e)}"

def ask_ai(ai_provider, client, model, prompt, log_file, max_retries=3, initial_delay=5):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            if ai_provider == 'ollama':
                return ask_ollama(model, prompt)
            elif ai_provider == 'openai':
                return ask_openai(client, model, prompt)
            elif ai_provider == 'anthropic':
                return ask_anthropic(client, model, prompt)
            else:
                write_to_log(log_file, "Invalid AI provider")
                return None
        except Exception as e:
            write_to_log(log_file, f"Error on attempt {attempt + 1}: {str(e)}")
            if "overloaded" in str(e).lower() or "rate limit" in str(e).lower():
                write_to_log(log_file, f"API overloaded or rate limited. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise
    write_to_log(log_file, "Max retries reached. Unable to get AI response.")
    return None

def get_ai_guess(ai_provider, client, model, ai_board_state, log_file, last_guess_result, previous_guesses, last_invalid_guess=None):
    prompt = (
        f"We're playing Battleship on an {BOARD_SIZE}x{BOARD_SIZE} grid. "
        f"Here are the ships hidden on the board:\n"
    )
    for ship, size in SHIPS:
        prompt += f"- {ship} ({size} squares)\n"

    prompt += f"\nHere's the current state of the board:\n"
    prompt += create_board_string(ai_board_state)
    prompt += "\n\n"

    if last_guess_result:
        prompt += f"Your last guess resulted in: {last_guess_result}\n\n"

    if last_invalid_guess:
        prompt += f"Your previous guess '{last_invalid_guess}' was invalid. Remember, the board is {BOARD_SIZE}x{BOARD_SIZE}.\n\n"

    prompt += "Your previous valid guesses were:\n"
    for guess in previous_guesses:
        prompt += f"{guess}, "
    prompt = prompt.rstrip(', ') + "\n\n"

    prompt += (
        f"Based on this information, what's your next guess? Explain your reasoning, "
        f"then provide your guess in the following format:\n\n"
        f"[REASONING]\nYour explanation here...\n[/REASONING]\n\n[GUESS]A5[/GUESS]\n\n"
        f"Remember, a valid guess is a letter (A-H) followed by a number (1-8)."
    )

    write_to_log(log_file, f"Human: {prompt}")

    ai_response = ask_ai(ai_provider, client, model, prompt, log_file)

    if ai_response is None:
        write_to_log(log_file, "Game: API error or no response from AI.")
        return None, None, "API error", None

    write_to_log(log_file, f"AI: {ai_response}")

    guess = extract_guess(ai_response)
    if guess:
        row, col = guess[0], int(guess[1])
        row_index = ord(row) - ord('A')
        col_index = col - 1
        if 0 <= row_index < BOARD_SIZE and 0 <= col_index < BOARD_SIZE:
            return row_index, col_index, None, None
        else:
            error_msg = f"Invalid guess {guess}: out of bounds for {BOARD_SIZE}x{BOARD_SIZE} board."
            write_to_log(log_file, f"Game: {error_msg}")
            return None, None, error_msg, guess
    else:
        error_msg = f"Could not extract a valid guess from AI response: '{ai_response}'."
        write_to_log(log_file, f"Game: {error_msg}")
        return None, None, error_msg, ai_response

def get_human_guess(stdscr):
    curses.echo()
    stdscr.addstr(BOARD_SIZE*3 + 5, 0, "Enter your guess (e.g., A5): ")
    guess = stdscr.getstr().decode('utf-8').strip().upper()
    curses.noecho()
    if guess == 'Q':
        return 'quit'
    coord = extract_coordinate(guess)
    if coord:
        row, col = coord
        return (ord(row) - ord('A'), col - 1)
    return None

def write_result_to_jsonl(ai_provider, ai_model, board_size, ships, turns, result, guesses, invalid_guesses):
    result_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": f"{ai_provider}:{ai_model}" if ai_provider and ai_model else "human",
        "board_size": board_size,
        "ships": ships,
        "turns": turns,
        "result": result,
        "guesses": guesses,
        "invalid_guesses": invalid_guesses
    }

    with open("results.jsonl", "a") as f:
        json.dump(result_data, f)
        f.write("\n")

    return result_data

def play_game(stdscr, ai_provider=None, ai_model=None):
    curses.curs_set(0)
    stdscr.clear()
    curses.mousemask(curses.ALL_MOUSE_EVENTS)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/battleship_log_{timestamp}"
    if ai_provider and ai_model:
        log_filename += f"_{ai_provider}_{ai_model}"
    log_filename += ".txt"
    log_file = open(log_filename, 'w')

    write_to_log(log_file, f"Game started at {timestamp}")
    if ai_provider and ai_model:
        write_to_log(log_file, f"Playing against AI: {ai_provider} - {ai_model}")
    else:
        write_to_log(log_file, "Human player mode")

    if ai_provider == 'openai':
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    elif ai_provider == 'anthropic':
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        client = None

    board, ai_board_state, ships_positions = initialize_game()
    ships_sunk = 0
    guesses = 0
    last_guess_result = None
    previous_guesses = []
    invalid_guess_count = 0
    api_error_count = 0
    last_invalid_guess = None
    game_interrupted = False
    game_result = None
    final_board = None

    try:
        while ships_sunk < len(SHIPS) and guesses < MAX_GUESSES:
            draw_board(stdscr, board)

            stdscr.addstr(BOARD_SIZE*3 + 2, 0, f"Guesses: {guesses}/{MAX_GUESSES} | Ships sunk: {ships_sunk}/{len(SHIPS)}")
            stdscr.addstr(BOARD_SIZE*3 + 3, 0, f"Invalid guesses: {invalid_guess_count} | API errors: {api_error_count}")
            if ai_provider and ai_model:
                stdscr.addstr(BOARD_SIZE*3 + 4, 0, f"AI ({ai_provider}:{ai_model}) is thinking... Press 'q' to quit.")
            else:
                stdscr.addstr(BOARD_SIZE*3 + 4, 0, "Enter your guess or press 'q' to quit.")

            stdscr.refresh()

            if ai_provider and ai_model:
                row, col, error_msg, invalid_guess = get_ai_guess(ai_provider, client, ai_model, ai_board_state, log_file, last_guess_result, previous_guesses, last_invalid_guess)
                if error_msg:
                    if error_msg == "API error":
                        api_error_count += 1
                        stdscr.addstr(BOARD_SIZE*3 + 5, 0, "API error occurred. Retrying...")
                    else:
                        invalid_guess_count += 1
                        last_invalid_guess = invalid_guess
                        stdscr.addstr(BOARD_SIZE*3 + 5, 0, f"AI made an invalid guess: {error_msg}")
                    stdscr.refresh()
                    curses.napms(2000)  # Pause for 2 seconds to show the message
                    continue
                cell = (row, col)
                last_invalid_guess = None
            else:
                cell = get_human_guess(stdscr)
                if cell == 'quit':
                    game_interrupted = True
                    break
                elif cell is None:
                    stdscr.addstr(BOARD_SIZE*3 + 6, 0, "Invalid guess. Try again.")
                    stdscr.refresh()
                    curses.napms(1000)
                    continue

            if cell:
                row, col = cell
                guess_coord = f"{chr(row + ord('A'))}{col + 1}"
                if guess_coord not in previous_guesses:
                    result = process_guess(board, ai_board_state, row, col, ships_positions, log_file)
                    if "sunk" in result:
                        ships_sunk += 1
                    guesses += 1
                    previous_guesses.append(guess_coord)
                    last_guess_result = f"{guess_coord}: {result}"
                else:
                    last_guess_result = f"{guess_coord}: Already guessed"



        if not game_interrupted:
            draw_board(stdscr, board, hide_ships=False)
            if ships_sunk == len(SHIPS):
                game_over_message = f"Game Over! All ships sunk in {guesses} guesses."
                game_result = "win"
            else:
                game_over_message = f"Game Over! Maximum guesses ({MAX_GUESSES}) reached. Ships sunk: {ships_sunk}/{len(SHIPS)}"
                game_result = "loss"

            stdscr.addstr(BOARD_SIZE*3 + 2, 0, game_over_message)
            write_to_log(log_file, game_over_message)
            write_to_log(log_file, f"Game ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            final_board = [row[:] for row in board]  # Create a deep copy of the board

    except KeyboardInterrupt:
        game_interrupted = True
        write_to_log(log_file, "Game interrupted by user.")
    finally:
        log_file.close()
        if game_interrupted:
            stdscr.addstr(BOARD_SIZE*3 + 2, 0, "Game interrupted. Press any key to exit.")
            stdscr.refresh()
            stdscr.getch()

    return game_result, guesses, previous_guesses, invalid_guess_count, final_board

def main(ai_provider=None, ai_model=None):
    result = curses.wrapper(play_game, ai_provider, ai_model)

    if result[0] is not None:  # Check if the game wasn't interrupted
        game_result, guesses, previous_guesses, invalid_guess_count, final_board = result

        # Write results to JSONL file and get the result data
        result_data = write_result_to_jsonl(ai_provider, ai_model, BOARD_SIZE, SHIPS, guesses, game_result, previous_guesses, invalid_guess_count)

        # Print JSONL and board to stdout
        print(json.dumps(result_data))
        print("\nFinal Board:")
        print(create_board_string(final_board))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Battleship against an AI or as a human.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ollama", metavar="MODEL", help="Use Ollama AI with the specified model")
    group.add_argument("--openai", metavar="MODEL", help="Use OpenAI with the specified model")
    group.add_argument("--anthropic", metavar="MODEL", help="Use Anthropic's Claude with the specified model")
    args = parser.parse_args()

    ai_provider = None
    ai_model = None

    if args.ollama:
        ai_provider = "ollama"
        ai_model = args.ollama
        if "OLLAMA_API_URL" not in os.environ:
            print("Warning: OLLAMA_API_URL not set. Using default 'http://localhost:11434'")
    elif args.openai:
        ai_provider = "openai"
        ai_model = args.openai
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable not set")
            sys.exit(1)
    elif args.anthropic:
        ai_provider = "anthropic"
        ai_model = args.anthropic
        if "ANTHROPIC_API_KEY" not in os.environ:
            print("Error: ANTHROPIC_API_KEY environment variable not set")
            sys.exit(1)

    main(ai_provider, ai_model)
