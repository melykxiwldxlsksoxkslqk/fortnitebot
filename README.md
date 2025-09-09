# EpicBot - Cloud Gaming Bot

This project is a bot that automates actions in games through the Microsoft Cloud Gaming (Xbox Cloud Gaming) service. It uses browser automation and computer vision to simulate a real user.

## How it works

The bot uses the following technologies:

- **Python**: The main programming language.
- **Playwright**: For browser automation (login, game launch).
- **OpenCV**: For computer vision to detect game elements on the screen.
- **PyAutoGUI**: To simulate mouse and keyboard inputs.
- **Stable-Baselines3**: For RL agent training (optional).

## Project Structure

- `app.py`: Desktop GUI to manage accounts, proxies, and run multiple bots.
- `assets/`: Contains image samples for OpenCV to find on the screen (e.g., buttons, icons).
- `config/`: Stores configuration files, like account credentials.
- `src/`: Contains the main source code for the bot.
- `requirements.txt`: A list of Python dependencies.

## Setup and Usage

1.  **Install dependencies:**
    ```bash
    python -m pip install -r requirements.txt
    ```
2.  **Install Playwright browsers:**
    ```bash
    python -m playwright install
    ```
3.  **Configure accounts:**
    -   Use the GUI to add accounts and save, or
    -   Create `config/accounts.txt` with lines in the format `login:password`.
4.  **Assets:**
    -   Replace placeholder `.png.txt` files by capturing screenshots and saving as `.png` with the same base name (e.g., `creative_mode_button.png`).
5.  **Run GUI:**
    ```bash
    python app.py
    ```
6.  **Run CLI (single-run training+play):**
    ```bash
    python -m src.main
    ``` 