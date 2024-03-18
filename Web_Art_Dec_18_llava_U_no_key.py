#JMR Web Art Maker updated  
#GUI July 3 2023 runs on GPUs on MAC!!
#Sept Added Wuerstchen logic and moved to autopipline
#Dec added from image ability
#Dec added llava for explanations
#Dec made univeral to run on ubutu with GPUs and Mac
#MPS having issues with llava and from image so set to CPU Dec 2023
#Dec 13 Working using Gradio
#July 8 added stable-diffusion-xl-base-0.9!
#Remember to update safety checkers each time you update
#When GPUs < 32 GB pip install bitsandbytes accelerate setuptools scipy 
# error if number of cuda devices  does not match.
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

import os
import torch
import re
import platform
import requests
from PIL import Image
from datetime import datetime
from io import BytesIO
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image
import platform
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
import subprocess
from transformers import AutoProcessor, LlavaForConditionalGeneration
import gradio as gr
from openai import OpenAI
import io

# OpenAI API Key
api_key = PUT YOUR KEY HERE   # pystaller needs to key here.
API_KEY = api_key
client = OpenAI(api_key=API_KEY)
print(API_KEY)  

lmodel_name = None

if platform.system() == 'Darwin':  # Darwin stands for macOS
    default_path = "/Users/jonathanrothberg/Diffusion_Models"
    art_directory = "/Users/jonathanrothberg/AIArt"
    llava_full_path =  "/Users/jonathanrothberg/llava-1.5-13b-hf" 

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


def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9\-\_\.]', '_', filename)


def llava(max_tokens, text, image):
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

    max_tokens = max_tokens
    prompt = text  # get the prompt from the text entry   
    prompt = "USER: <image>\n"+ prompt +"\nASSISTANT:" #refer to images using the special <image> token. To indicate which text comes from a human vs. the model, one uses USER and ASSISTANT respectively. 
    print (prompt) 
    inputs = processor(prompt, image, return_tensors='pt').to(llava_gpu)
    #print (inputs)
    output = lmodel.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    answer = (processor.decode(output[0], skip_special_tokens=True))
    print (answer)

    return answer


def generate_images(from1, input_text, model_name, num_drawings, guidance_scale, strength, num_inference_steps, image):
    
    model_path = os.path.join(default_path, model_name)
    
    num_drawings = num_drawings
    guidance_scale = guidance_scale
    num_inference_steps = num_inference_steps
    print (input_text)
    prompt = input_text  # get the prompt from the text entry
    ouput_text = "Enjoy your art!"

    if model_name != "DaleE":
        
        if from1 == "image":
            if image is not None:
                print("Image Included")

                pipe = AutoPipelineForImage2Image.from_pretrained(model_path, torch_dtype=dtype ) #float16 gave error
                pipe = pipe.to(device_type) # when you don't do this it runs on CPUs. Lots of them!
                pipe.enable_attention_slicing()

                image = pipe(
                    prompt=prompt,
                    image=image, 
                    height=1024,
                    width=1024,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_drawings,
                ).images

        else:
            pipe = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=dtype)
            pipe = pipe.to(device_type)
            pipe.enable_attention_slicing()

            if model_name != "wuerstchen":
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
            #image[i].show(title=file_name)
            print (file_name)

    else: # DaleE - getting API key from environment variable if you don't set it
        image = []
        for i in range(num_drawings):
            try:
                response = client.images.generate(model="dall-e-3",prompt=prompt, size="1024x1024", quality="standard",n=1)
                image_url = response.data[0].url
                #print(image_url)
                # Get the image
                response = requests.get(image_url)
                # Convert it to a PIL image object
                image_d = Image.open(io.BytesIO(response.content))
                # Display the image
                #image_d.show()
                short_name = prompt[:15]
                timestamp = datetime.now().strftime("%f")
                timestamp = timestamp[-4:]
                file_name = sanitize_filename(f"{short_name}_{model_name}_{i+1}_{timestamp}.png")
                image_d.save(os.path.join(image_dir, file_name))
                image.append(image_d)
            except Exception as e:
                print ("DaleE error",e)
                ouput_text = "DaleE censorship error. Try again."
    return image, ouput_text


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

print ("Directory: ", default_path)
Diff_dir = [name for name in os.listdir(default_path) if os.path.isdir(os.path.join(default_path, name))] # only directories
print("Available models:")
for i, file in enumerate(Diff_dir, 1):
    print(f"{i}. {file}")

Diff_dir = Diff_dir + ["DaleE"]
print("Diff_dir", Diff_dir)


def talk_to_functions(input_text, model_name, num_drawings, guidance_scale, strength, num_inference_steps, max_new_tokens, action, gradio_image):
    output_text = ""
    #output_image = None
    output_images = []
    # Convert 'gradio_image' to PIL image if it's not None
    image = Image.fromarray(gradio_image) if gradio_image is not None else None

    model_name = Diff_dir[0] if model_name == [] else model_name # set default model
    print (model_name)
    action = "Generate From Text" if action == None else action # set default action
    input_text = "A Rottweiler Puppy" if input_text == "" else input_text # set default text
    print (action, input_text)
    # Convert string inputs to integers
    num_drawings = int(num_drawings)
    num_inference_steps = int(num_inference_steps)
    max_new_tokens = int(max_new_tokens)

    # Check for missing inputs
    if action in ["Generate From Text", "Generate From Image & Prompt"] and not model_name:
        output_text = "No model selected."
    elif action == "Generate From Text" and not input_text:
        output_text = "No text provided."
    elif action == "Generate From Image & Prompt" and image is None and model_name != "DaleE":
        output_text = "No image provided."
    elif action == "Generate From Image & Prompt" and model_name == "DaleE":
        output_text = "DaleE Currently Only Generates from Text."
    elif action == "Tell Me About Image" and image is None:
        output_text = "No image provided."
    elif action not in ["Generate From Text", "Generate From Image & Prompt", "Tell Me About Image"]:
        output_text = "No selection made."

    # If there's an error message, print it and return early
    if output_text:
        print(output_text)
        return output_text, output_images

    # Perform actions based on the selected action
    if action == "Generate From Text":
        print("Generating from text.")
        output_image, output_text = generate_images(from1="text", input_text=input_text, model_name=model_name, num_drawings=num_drawings, guidance_scale=guidance_scale, strength=strength, num_inference_steps=num_inference_steps, image=image)

    elif action == "Generate From Image & Prompt":
        print("Generating from image and prompt.")
        old_strength = strength
        if strength * num_inference_steps < 1:
            # Adjust strength
            strength = 1 / num_inference_steps
            print("adjusting strength")
        print("Generating from image")
        output_image, output_text = generate_images(from1="image", input_text=input_text, model_name=model_name, num_drawings=num_drawings, guidance_scale=guidance_scale, strength=strength, num_inference_steps=num_inference_steps, image=image)
        if old_strength != strength:
            output_text = f"The product of Strength and Number of interference steps was less than 1. Strength has been adjusted to {strength}."

    elif action == "Tell Me About Image":
        print("Tell me about image")
        output_text = llava(max_tokens=max_new_tokens, text=input_text, image=image)
        output_image = []

    if isinstance(output_image, list): #maybe redundant since made all lists
        output_images.extend(output_image)  # Extend the list with all images
    else:
        output_images.append(output_image)  # Append the single image to the list

    return output_text, output_images  # Return the list of images


web_interface = gr.Interface(
    fn=talk_to_functions, 
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Ask me something interesting or I will draw a Rottweiler puppy."),
        gr.Dropdown(Diff_dir, label="Select Model to Generate Art or I'll use " + Diff_dir[0]),  
        gr.Slider(1, 100, 2, label="Number of Drawings", step=1), 
        gr.Slider(0.01, 20, 7.5, label="Guidance Scale"),
        gr.Slider(0.04, 1, 0.5, label="Strength - Lower is More Like Image"),
        gr.Slider(1, 100, 25, label="Number of Inference Steps", step=1),
        gr.Slider(1, 4096, 512, label="Max New Tokens", step=1),

        
        gr.Radio(["Generate From Text", "Generate From Image & Prompt", "Tell Me About Image"], label="Action"),
        gr.Image(label="Upload Image")
    ],
    outputs=[
        gr.Textbox(label="Output Text"),
        gr.Gallery(label="Output Images")  # Use Gallery to display a list of images
        #gr.Image(label="Output Image")
    ],
    title="JMR's Art Maker with Llava for Explanations",
    description= """This is a web based chatbot that can generate and explain images. 
    Select a model and provide either a prompt or paste in an image. 
    Pick an action. 
    The model will generate images based on the Prompt and or Image (DaleE is text driven only for now).
    You can also ask the model to explain an image.
    You can save the images or even drag the into upload do modify them. The models except for DaleE run locally on my computer."""
)
try:
    web_interface.launch(share=True)
except:
    web_interface.launch(share=False)