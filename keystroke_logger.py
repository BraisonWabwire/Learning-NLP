from pynput import keyboard
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# List to store keystrokes
keystrokes = []

def on_press(key):
    try:
        # Normal alphanumeric key
        keystrokes.append(f"{key.char}")
    except AttributeError:
        # Special key
        keystrokes.append(f"[{key}]")

def on_release(key):
    if key == keyboard.Key.esc:
        # Save to PDF on ESC key
        save_to_pdf(keystrokes)
        return False

def save_to_pdf(keystrokes):
    filename = f"keystrokes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    text_obj = c.beginText(40, height - 40)
    text_obj.setFont("Courier", 12)

    text_obj.textLine("Keystroke Log:")
    text_obj.textLine("-" * 50)

    # Combine all keystrokes into lines of reasonable length
    buffer = ""
    for key in keystrokes:
        if len(buffer) > 80:
            text_obj.textLine(buffer)
            buffer = ""
        buffer += key + " "
    if buffer:
        text_obj.textLine(buffer)

    c.drawText(text_obj)
    c.save()
    print(f"[✔] Keystrokes saved to {filename}")

# Start the listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
