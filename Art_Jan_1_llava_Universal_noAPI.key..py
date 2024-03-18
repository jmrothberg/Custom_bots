#JMR GUI Art Maker updated  GUI July 3 2023 runs on GPUs on MAC!!
# to use Open Dalle Dec 26 2023 
# scheduling_k_dpm_2_ancestral_discrete.py", line 279, in set_timesteps
# ADD This line: timesteps = timesteps.astype('float32')
# timesteps = torch.from_numpy(timesteps).to(device)
#July 8 added stable-diffusion-xl-base-0.9!
#Sept Added Wuerstchen logic and moved to autopipline
#Dec added from image ability
#Remember to update safety checkers each time you update
#'/home/jonathan/GPTBots/.venv/lib/python3.10/site-packages/diffusers/pipelines/stable_diffusion/safety_checker.py' 
#print ("No censors")
        #for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
        #    if has_nsfw_concept:
        #       if torch.is_tensor(images) or torch.is_tensor(images[0]):
        #           images[idx] = torch.zeros_like(images[idx])  # black image
        #       else:
        #          images[idx] = np.zeros(images[idx].shape)  # black image

        #if any(has_nsfw_concepts):
        #    logger.warning(
        #        "Potential NSFW content was detected in one or more images. A black image will be returned instead."
        #       " Try again with a different prompt and/or seed."
        #   )
#Check the system platform

import os
import torch
import re
import platform
import time
import requests
from PIL import Image
from datetime import datetime
from openai import OpenAI
from io import BytesIO
from base64 import b64decode
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import Menu, END
import tkinter.font as font 
from PIL import Image, ImageTk, ImageGrab
import platform
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import subprocess
from transformers import AutoProcessor, LlavaForConditionalGeneration
import io

# OpenAI API Key
API_KEY = "put your key here"   # pystaller needs to key here.
client = OpenAI(api_key=API_KEY)
print(platform.machine())
#API_KEY = os.getenv('OPENAI_API_KEY')                                        
print(API_KEY)  

lmodel_name = None
print ("lmodel_name", lmodel_name)
if platform.system() == 'Darwin':  # Darwin stands for macOS
    default_path = "/Users/jonathanrothberg/Diffusion_Models"
    art_directory = "/Users/jonathanrothberg/AIArt"
    llava_full_path =  "/Users/jonathanrothberg/llava-1.5-13b-hf" 
    #llava_full_path =  "/Users/jonathanrothberg/LLaVA-3b" #Not right format

else:
    default_path = "/data/Diffusion_Models"
    art_directory = "/data/AIArt"
    llava_full_path = "/data/llava-1.5-13b-hf"

print(f'Model Director: {default_path}. Location for art_directory: {art_directory}')

if torch.backends.mps.is_available():
    device_type = torch.device("mps")
    x = torch.ones(1, device=device_type)
    print (x)
    dtype = torch.float32
    dtype_llava = torch.float16
    llava_gpu = "cpu" # for llava until mps support
else:
    print ("MPS device not found. Going to CUDA")
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" #do yo need for bitandbytes?
    device_type = "cuda:1" #this way keep the diffusor models on GPU 1
    dtype = torch.float16
    llava_gpu = "cuda:0" # for llava bitsandbytes sends to gpu 0
    # Get the ID of the default device
    device_id = torch.cuda.current_device()
    print (f"Device ID: {device_id}")
    # Get the total memory of the GPU
    total_mem = torch.cuda.get_device_properties(device_id).total_memory
    # Convert bytes to GB
    total_mem_gb = total_mem / (1024 ** 3)
    print (f"Total memory: {total_mem_gb} GB")
    # Set 8bit flag
    is_8bit = total_mem_gb < 32
    print(f"8bit flag is set to {is_8bit}")
print ("Device type: ", device_type)


def show_context_menu(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.configure(font=default_font)  
    context_menu.add_command(label="Cut", command=lambda: root.focus_get().event_generate("<<Cut>>"))
    context_menu.add_command(label="Copy", command=lambda: root.focus_get().event_generate("<<Copy>>"))
    context_menu.add_command(label="Paste", command=lambda: root.focus_get().event_generate("<<Paste>>"))
    context_menu.add_command(label="Clear", command=lambda: clear_text(root.focus_get()))
    context_menu.tk_popup(event.x_root, event.y_root)


def show_context_menu_image(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.configure(font = default_font)
    context_menu.add_command(label="Paste Clipboard", command=lambda: paste_image_from_clipboard())
    context_menu.tk_popup(event.x_root, event.y_root)  


def clear_text(widget):
    if widget.winfo_class() == 'Text':
        widget.delete('1.0', END)
    elif widget.winfo_class() == 'Entry':
        widget.delete(0, END)


def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)


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


'''def paste_image_from_clipboard(event=None):
    try:
        # Use 'xclip' to read the PNG image data from the clipboard and save it to a BytesIO object
        png_data_bytes = subprocess.run(['xclip', '-selection', 'clipboard', '-t', 'image/png', '-o'], capture_output=True).stdout
        image_data = BytesIO(png_data_bytes)
        # Load the image from the BytesIO object
        image = Image.open(image_data)
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
        print("Error in paste_image_from_clipboard:", e)'''


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
    global global_image_reference, original_image, image_raw  # Declare the use of the global variables
    # Load the image using PIL
    image = Image.open(file_path)
    print ("file_path", file_path )
    original_image = image
    photo = ImageTk.PhotoImage(image)
    # If an image is already displayed, remove it
    for widget in drop_frame.winfo_children():
        widget.destroy()
    # Display the image in the drop area
    image_label = tk.Label(drop_frame, image=photo)
    image_label.image = photo  # Keep a reference to the image
    image_label.pack()


def clear_image_area():
    global global_image_reference, original_image  # Declare the use of the global variable
    for widget in drop_frame.winfo_children():
            widget.destroy()
    original_image = None


def llava():
    global lmodel_name, processor, lmodel
    print ("llava")
    if not lmodel_name:
        lmodel_name = llava_full_path
        
        if llava_gpu == "cuda:0" and is_8bit:
            lmodel = LlavaForConditionalGeneration.from_pretrained(
                        lmodel_name, 
                        torch_dtype=dtype,
                        load_in_8bits=is_8bit, 
                    )
            print ("model loaded, load_in_8bit", lmodel_name, is_8bit)
        else:
            lmodel = LlavaForConditionalGeneration.from_pretrained(
                        lmodel_name, 
                        torch_dtype=dtype,
                    ).to(llava_gpu)
    
        processor = AutoProcessor.from_pretrained(lmodel_name)
        print ("processor loaded")

    max_tokens = max_tokens_slider.get()
    prompt = prompt_entry.get('1.0', 'end-1c')  # get the prompt from the text entry   
    prompt = "USER: <image>\n"+ prompt +"\nASSISTANT:" #refer to images using the special <image> token. To indicate which text comes from a human vs. the model, one uses USER and ASSISTANT respectively. 
    print (prompt) 
    inputs = processor(prompt, original_image, return_tensors='pt').to(llava_gpu)
    #print (inputs)
    output = lmodel.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    answer = (processor.decode(output[0], skip_special_tokens=True))
    print (answer)
    answer_text.delete('1.0', tk.END)
    answer_text.insert(tk.END, answer)
    answer_text.update()
    answer_text.see(tk.END)


def generate_images(from1):
    # Get values from the GUI
    model_selection = [model_id for model_id, var in zip(model_ids, model_ids_vars) if var.get() == 1]
    print ("model_selection", model_selection)
    prompt = prompt_entry.get('1.0', 'end-1c')  # get the prompt from the text entry
    print (prompt)
    num_drawings = num_drawings_slider.get()
    guidance_scale = guidance_scale_slider.get()
    num_inference_steps = num_inference_steps_slider.get()
    strength = strength_slider.get()

    for model_id in model_selection:
        model_name = model_id[-10:]
        print (model_id)

        if model_id != "DaleE": 
            if from1 == "image":
                if original_image is not None:
                    print("Image Included")
                    #base64_image = global_image_reference # Used for OpenAI GPT4v

                    pipe = AutoPipelineForImage2Image.from_pretrained(os.path.join (path_local,model_id ), torch_dtype=torch.float32 ) #float16 gave error
                    
                    pipe = pipe.to(device_type) # when you don't do this it runs on CPUs. Lots of them!

                    image = pipe(
                        prompt=prompt,
                        image=original_image, 
                        num_inference_steps=num_inference_steps,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=num_drawings,
                    ).images
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(os.path.join (path_local,model_id ), torch_dtype=dtype)
                pipe = pipe.to(device_type)
                pipe.enable_attention_slicing()

                if model_id != "wuerstchen":
                    image = pipe(prompt = prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,num_images_per_prompt=num_drawings).images
                else:
                    image = pipe(
                        prompt=prompt,
                        height=1024,
                        width=1024,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        decoder_guidance_scale=guidance_scale,
                        num_images_per_prompt=num_drawings,
                    ).images
                    
            for i in range(num_drawings):    
                short_name = prompt[:15]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                file_name = sanitize_filename(f"{short_name}_{model_name}_{i+1}_{timestamp}.png")
                image[i].save(os.path.join(image_dir, file_name))
                image[i].show(title=file_name)
                print (file_name)
        else:
            size = "1024x1024"
            quality = "standard"
            for i in range(num_drawings):
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
                    answer_text.insert(tk.END, "DaleE error", e)
                    answer_text.insert(tk.END, "\n")
                    answer_text.update()
                    answer_text.see(tk.END)


if not os.path.exists(default_path):
    messagebox.showinfo("Information","Select Directory for Stable diffusion models") #crashes my mac if no root window.
    path_local = filedialog.askdirectory(title="Select Directory for Stable diffusion models")
else:
    path_local = default_path
print (path_local)

# Check if the default path with the model exists
if not os.path.exists(os.path.join (default_path, art_directory)):
    messagebox.showinfo("Information","Select Directory to save your Art")
    image_dir = filedialog.askdirectory(title="Select Directory to save your Art")
else:
    image_dir = art_directory
print (image_dir)

directories = (os.listdir(path_local))
model_ids = directories + ["DaleE"]
print(model_ids)

unix_system = platform.system()
print (unix_system)

if unix_system == "Linux":
    from tkinterdnd2 import TkinterDnD, DND_FILES
    root = TkinterDnD.Tk()
    print ("Ubuntu & tkinterdnd2")
else:
    root = tk.Tk()

root.title("JMR's Text or Image to Drawing with Llava for Explanations")
root.geometry("870x800")

# Create a frame for center alignment
frame = tk.Frame(root)
frame.pack()
default_font = tk.font.nametofont("TkDefaultFont")                           
default_font.configure(size=12)  

# Create the prompt entry
prompt_label = tk.Label(frame, text="\n Style and description of the Art you want to produce:")
prompt_label.pack()

prompt_entry = tk.Text(frame, height=4, width=105, wrap="word")  # height=2 makes it two lines high
prompt_entry.configure(font=default_font)
prompt_entry.pack()

prompt_entry.bind("<Button-2>", show_context_menu) #for macos
prompt_entry.bind("<Button-3>", show_context_menu) #for linux

# Create the answer output
answer_label = tk.Label(frame, text="\n Let me tell you about the image:")
answer_label.pack()

# Create the answer output with a vertical scrollbar
answer_text = tk.Text(frame, height=10, width=105, wrap="word")  # height=4 makes it two lines high
answer_text.configure(font=default_font)
answer_text.pack()

answer_text.bind("<Button-2>", show_context_menu) #for macos
answer_text.bind("<Button-3>", show_context_menu) #for linux

# Create the model selection checkboxes
model_ids_vars = [tk.IntVar() for _ in range(len(model_ids))]
for i in range(len(model_ids)):
    model_checkbox = tk.Checkbutton(frame, text=model_ids[i], variable=model_ids_vars[i], anchor = 'w')
    model_checkbox.pack()

# Create a frame for the sliders
sliders_frame = tk.Frame(frame)    
sliders_frame.pack(pady=10)  # pack sliders_frame

# Create the sliders within the sliders_frame with default values and pack them
num_drawings_slider = tk.Scale(sliders_frame, from_=1, to=1000, orient="horizontal", label="Number of Drawings", length=200)
num_drawings_slider.set(3)  # default value
num_drawings_slider.pack(side="left")  # pack num_drawings_slider

guidance_scale_slider = tk.Scale(sliders_frame, from_=1, to=20, orient="horizontal", label="Guidance Scale", length=200, resolution=0.1)
guidance_scale_slider.set(7.5)  # default value
guidance_scale_slider.pack(side="left")  # pack guidance_scale_slider

strength_slider = tk.Scale(sliders_frame, from_=0, to=1, orient="horizontal", label="Strength", length=200, resolution=0.1)
strength_slider.set(0.5)  # default value
strength_slider.pack(side="left")  # pack strength_slider

num_inference_steps_slider = tk.Scale(sliders_frame, from_=1, to=100, orient="horizontal", label="Inference Steps", length=200)
num_inference_steps_slider.set(25)  # default value
num_inference_steps_slider.pack(side="left")  # pack num_inference_steps_slider

max_tokens_slider = tk.Scale(frame, from_=8, to=4096, orient="horizontal", label="Max New Tokens",length=300)
max_tokens_slider.set(256)  # default value
max_tokens_slider.pack()

# Create a frame for the buttons
buttons_frame = tk.Frame(frame)
buttons_frame.pack(pady=10)  # pack buttons_frame

# Create the buttons within the buttons_frame and pack them
generate_text_button = tk.Button(buttons_frame, text="Generate From Text", command=lambda: generate_images(from1="text"))
generate_text_button.pack(side="left", padx=5)  # pack generate_text_button

generate_image_button = tk.Button(buttons_frame, text="Generate From Image & Text", command=lambda: generate_images(from1="image"))
generate_image_button.pack(side="left", padx=5)  # pack generate_image_button

llava_button = tk.Button(buttons_frame, text="Tell Me About Image (llava)", command=llava)
llava_button.pack(side="left", padx=5)  # pack llava_button

paste_button = tk.Button(buttons_frame, text="Paste From Cipboard", command=paste_image_from_clipboard)
paste_button.pack(side="left", padx=5)  # pack paste_button

clear_button = tk.Button(buttons_frame, text="Clear image", command=clear_image_area)
clear_button.pack(side="left", padx=5)  # pack clear_button

# Create a third frame for the drag & drop area
drop_frame_container = tk.Frame(root)
drop_frame_container.pack(fill='both', expand=True, padx=10, pady=10)

# Create a label for the image directory
image_dir_label = tk.Label(root, text="Drawings are saved in "+ image_dir)
image_dir_label.pack(side="bottom")

# Create the drop_frame label within the third frame
drop_frame = tk.Label(drop_frame_container, text="Paste, Drag & drop (unix) image file here", bg="lightgrey", width=80, height=10)
drop_frame.pack(fill='both', expand=True)

drop_frame.bind("<Button-2>", show_context_menu_image)
drop_frame.bind("<Button-3>", show_context_menu_image)

if unix_system == "Linux":
    # Register the drag & drop handler
    drop_frame.drop_target_register(DND_FILES)
    drop_frame.dnd_bind('<<Drop>>', drop)

# Adjust the image_dir_label to pack at the bottom
image_dir_label.pack(side="bottom", fill='x')

# Adjust the grid configuration for better alignment
frame.grid_columnconfigure(0, weight=1)  # This ensures that the column expands to fill any extra space

# Start the main loop
root.mainloop()