#JMR OpenAI GUI Select & Temp SimpleChat with Streaming July 5 2023  # cut and paste to the menu!
#JMR November added ability to look at images.
#Sets text sized
#JMR November added ability to look at images for mac, pi, and ubuntu.
#But drag and drop only works on ubuntu. And no tkinterdnd2 on mac, and clip had errors on mac.

import os
import tkinter as tk
from tkinter import scrolledtext
import time
import platform
from openai import OpenAI
from tkinter import Menu, END
import tkinter.font as font 
from PIL import Image, ImageTk
import base64
import requests
import io
import re
from io import BytesIO
from tkinter import filedialog
import tempfile
import subprocess
from PIL import Image, ImageTk
from base64 import b64decode
from datetime import datetime
from PIL import ImageGrab

# OpenAI API Key
API_KEY = "put your key here"   # pystaller needs to key here.
client = OpenAI(api_key=API_KEY)
print(platform.machine())

#API_KEY = os.getenv('OPENAI_API_KEY')   
print(API_KEY)                                                              

api_key = API_KEY # Used by OpenAI API for GPT calls.  

#model_name  = "gpt-3.5-turbo"
model_name = "gpt-4-vision-preview"
temp = 0

# Define a global variable to store the image reference
global_image_reference = None
original_image = None

image_dir = "created_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def clear_text(widget):
    if widget.winfo_class() == 'Text':
        widget.delete('1.0', END)
    elif widget.winfo_class() == 'Entry':
        widget.delete(0, END)

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)

def show_context_menu(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.configure(font = default_font)
    context_menu.add_command(label="Cut", command=lambda: root.focus_get().event_generate("<<Cut>>"))
    context_menu.add_command(label="Copy", command=lambda: root.focus_get().event_generate("<<Copy>>"))
    context_menu.add_command(label="Paste", command=lambda: root.focus_get().event_generate("<<Paste>>"))
    context_menu.add_command(label="Clear", command=lambda: clear_text(root.focus_get()))
    #context_menu.add_separator()
    #context_menu.add_command(label="Paste Clipboard", command=lambda: paste_image_from_clipboard())
    context_menu.tk_popup(event.x_root, event.y_root)  

def show_context_menu_image(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.configure(font = default_font)
    context_menu.add_command(label="Paste Clipboard", command=lambda: paste_image_from_clipboard())
    context_menu.tk_popup(event.x_root, event.y_root)  

def set_model_name(name):
    global model_name
    model_name = name
    temp = tempLLM.get()
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window

def drop(event):
    file_path = event.data
    if file_path.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Here you can load the image and do something with it
        print(f"File dropped: {file_path}")
        # For example, display the image using a Label
        load_and_display_image(file_path)
    else:
        print("Not a valid image file.")


def load_and_display_image(file_path):
    global global_image_reference, original_image  # Declare the use of the global variables
    # Load the image using PIL
    print ("loading and displaying image", file_path)
    image = Image.open(file_path)
    original_image = image

    # Convert the image to bytes and then encode it
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    bphoto = base64.b64encode(img_byte).decode('utf-8')
    # Store the image in the global variable
    global_image_reference = bphoto

    photo = ImageTk.PhotoImage(image)

    # If an image is already displayed, remove it
    for widget in drop_frame.winfo_children():
        widget.destroy()

    # Display the image in the drop area
    image_label = tk.Label(drop_frame, image=photo)
    image_label.image = photo  # Keep a reference to the image
    image_label.pack()


def save_text():
    prompt = entry.get("1.0", "end-1c")  # Get text from Text widget
    generated_text = text_area.get("1.0", "end-1c")  # Get text from Text widget

    # Create the filename using the first 10 characters of the prompt and a 4-digit timestamp
    filename = prompt[:10] + "_" + time.strftime("%m%d-%H%M") + ".txt"

    # Create a directory to save the files if it doesn't exist
    directory = "saved_texts"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the prompt and generated text to the file
    with open(os.path.join(directory, filename), "w") as file:
        file.write("Prompt:\n" + prompt + "\n\nGenerated Text:\n" + generated_text)
    print("Text saved successfully in: ",filename)


def save_image():
    if original_image:  # Check if the original image exists
        # Create a directory to save the images if it doesn't exist
        directory = "saved_images"
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Create the filename using a 4-digit timestamp
        filename = "image_" + time.strftime("%m%d-%H%M") + ".jpeg"

        # Save the original image to the file
        original_image.save(os.path.join(directory, filename), "JPEG")
        print("Image saved successfully in:", filename)


def paste_image_from_clipboard(event=None):
    try:
        # Use 'ImageGrab' to read the image data from the clipboard
        image = ImageGrab.grabclipboard()
        
        # If image is in RGBA (with alpha channel), convert to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Generate a temporary path to save the image locally (optional, only if you need to save the file)
        temp_image_path = f"/tmp/temp_image_{int(time.time())}.png"
        image.save(temp_image_path)

        # Load and display the image using your existing function
        load_and_display_image(temp_image_path)
    except Exception as e:
        # If there is an error, print the error message
        print("Error in paste_image_from_clipboard:", e)


def clear_image_area():
    global global_image_reference  # Declare the use of the global variable
    for widget in drop_frame.winfo_children():
            widget.destroy()
    global_image_reference = None
    drop_frame.configure(text="Paste Image Here")

def draw_image():
    size = size_var.get()
    quality = quality_var.get()
    model_name = "dall-e-3" 
    print(f"Drawing an image with size: {size}, quality: {quality}, using model: {model_name}")

    prompt = entry.get("1.0", "end-1c") # get text from Text widget
    
    try:
        response = client.images.generate(model="dall-e-3",prompt=prompt, size=size, quality=quality,n=1)
        image_url = response.data[0].url
        #print(image_url)
        # Get the image
        response = requests.get(image_url)
        # Convert it to a PIL image object
        image_d = Image.open(io.BytesIO(response.content))
        # Display the image
        image_d.show()
        short_name = prompt[:15]
        timestamp = datetime.now().strftime("%f")
        timestamp = timestamp[-4:]
        file_name = sanitize_filename(f"{short_name}_{model_name}_{timestamp}.png")
        image_d.save(os.path.join(image_dir, file_name))
    
    except Exception as e:
        print ("DaleE error", e)
        text_area.insert(tk.END, "DaleE error", e)
        text_area.insert(tk.END, "\n")
        text_area.update() 


def talk_to_LLM():
    prompt = entry.get("1.0", "end-1c") # get text from Text widget
    
    if global_image_reference is not None:
        print("Image Included")
    base64_image = global_image_reference

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }   

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": int(max_new_tokens.get())
    }

    if base64_image is not None and model_name == "gpt-4-vision-preview":
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        payload["messages"][0]["content"].append(image_content)

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    #print (response) # a function when streaming.
    response = response.json()
    #print (response)
    if 'choices' in response and len(response['choices']) > 0:
        answer = (response['choices'][0]['message']['content'])
    elif 'error' in response:
        answer = (response['error']['message'])
    else:
        answer = "No choices or error message found in the response."
        
    print (answer)  
    text_area.insert(tk.END, f"\nResult: ")
    text_area.insert(tk.END, answer)
    text_area.insert(tk.END, "\n")
    text_area.update() 
    text_area.see(tk.END)

unix_system = platform.system()
print (unix_system)

if unix_system == "Linux":
    from tkinterdnd2 import TkinterDnD, DND_FILES
    root = TkinterDnD.Tk()
    print ("Ubuntu & tkinterdnd2")
else:
    root = tk.Tk()

default_font = tk.font.nametofont("TkDefaultFont")                           
default_font.configure(size=10)  

root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window
root.geometry("1100x600")
root.grid_columnconfigure(0, weight=1) # Column 0 takes all available horizontal space

entry = scrolledtext.ScrolledText(root, height=5, wrap="word")
entry.configure(font = default_font)
entry.grid(row=0, column=0, columnspan=16, sticky='nsew')
root.grid_rowconfigure(0, weight=2, minsize=100)  

entry.bind("<Button-2>", show_context_menu)
entry.bind("<Button-3>", show_context_menu)

save_button = tk.Button(root, text="Save", command=save_text)
save_button.grid(row=1, column=0, sticky='w')

# Define the control variables for image size and quality buttons
size_var = tk.StringVar(value='1024x1024')  # Default value
quality_var = tk.StringVar(value='standard')  # Default value

radio_1024x1024 = tk.Radiobutton(root, text="1024", variable=size_var, value='1024x1024')
radio_1024x1024.grid(row=1, column=1, sticky='w', padx=(0,0))

radio_1024x1792 = tk.Radiobutton(root, text="1024x1792", variable=size_var, value='1024x1792')
radio_1024x1792.grid(row=1, column=1, sticky='w', padx=(50,0))  # Adjust the padding as needed

radio_1792x1024 = tk.Radiobutton(root, text="1792x1024", variable=size_var, value='1792x1024')
radio_1792x1024.grid(row=1, column=1, sticky='w', padx=(130,0))  # Adjust the padding as needed

radio_standard = tk.Radiobutton(root, text="Standard", variable=quality_var, value='standard')
radio_standard.grid(row=1, column=2, sticky='w',padx=(0,10))

radio_hd = tk.Radiobutton(root, text="HD", variable=quality_var, value='hd')
radio_hd.grid(row=1, column=2, sticky='w', padx=(70,0))  # Adjust the padding as needed

# Define temperature and max tokens sliders
temp_label = tk.Label(root, text="Temp:")
temp_label.grid(row=1, column=5, sticky='w')

tempLLM = tk.DoubleVar(value = 0.7)
slider = tk.Scale(root, from_=0, to=2, resolution=0.01, orient=tk.HORIZONTAL, variable=tempLLM)
slider.grid(row=1, column=6, sticky='w')
temp = tempLLM.get()

max_label = tk.Label(root, text="Tokens:")
max_label.grid(row=1, column=7, sticky='w')

max_new_tokens = tk.DoubleVar(value = 256)
slider_token = tk.Scale(root, from_=8, to=4096, resolution=8, orient=tk.HORIZONTAL, variable=max_new_tokens)
slider_token.grid(row=1, column=8, sticky='w')

# Variable to hold the selected model name
model_var = tk.StringVar()

# Define the radio buttons for the different models
radio_3_5 = tk.Radiobutton(root, text="GPT4", variable=model_var, value="gpt-4", command=lambda: set_model_name(model_var.get()))
radio_3_5.grid(row=1, column=9, columnspan=1)

radio_4T = tk.Radiobutton(root, text="GPT4T", variable=model_var, value="gpt-4-turbo-preview", command=lambda: set_model_name(model_var.get()))
radio_4T.grid(row=1, column=10, columnspan=1)

radio_4V = tk.Radiobutton(root, text="GPT4V", variable=model_var, value="gpt-4-vision-preview", command=lambda: set_model_name(model_var.get()))
radio_4V.grid(row=1, column=11, columnspan=1)

# Optionally, set a default value
model_var.set("gpt-3.5-turbo")

button = tk.Button(root, text="Prompt", command=talk_to_LLM)
button.grid(row=1, column=12, sticky='e')

image_button = tk.Button(root, text="Draw", command=draw_image)
image_button.grid(row=1, column=13, sticky='w')

paste_button = tk.Button(root, text="Paste", command=paste_image_from_clipboard)
paste_button.grid(row=1, column=14, sticky='w')

clear_button = tk.Button(root, text="Clear", command=clear_image_area)
clear_button.grid(row=1, column=15, sticky='w')  # Adjust the column index if needed

text_area = tk.scrolledtext.ScrolledText(root, height=7, wrap="word")
text_area.configure(font=default_font)
text_area.grid(row=2, column=0, columnspan=16, sticky='nsew')
root.grid_rowconfigure(2, weight=5, minsize=100)  

text_area.bind("<Button-2>", show_context_menu)
text_area.bind("<Button-3>", show_context_menu)

drop_frame = tk.Label(root, text="Paste Image Here", bg="lightblue", width=80, height=10)
drop_frame.grid(row=3, column=0, columnspan=16, padx=10, pady=10, sticky='nsew')
root.grid_rowconfigure(3, weight=1, minsize=20)  

drop_frame.bind("<Button-2>", show_context_menu_image)
drop_frame.bind("<Button-3>", show_context_menu_image)

if unix_system == "Linux":
    # Register the drag & drop handler
    drop_frame.drop_target_register(DND_FILES)
    drop_frame.dnd_bind('<<Drop>>', drop)

root.mainloop()