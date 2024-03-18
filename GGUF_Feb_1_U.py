#JMR SimpleChat buttons for models slider for temp  Streaming July 3 2023 
#Use ask directory so do not need to know the director in advance
#export LLAMA_METAL=on
#echo $LLAMA_METAL
#CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
#export LLAMA_CUBLAS=1
#pip install sentence_transformers
#echo $LLAMA_CUBLAS
#CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -U llama-cpp-python --no-cache-dir
#August 20 updated for new Chromadb format. Had to rewrite.
#Got it all working, but had to delete the .sql files in datas should uses the local metal but did find it so used the botenv.
#python3 convert-llama-ggmlv3-to-gguf.py -i /Users/jonathanrothberg/WorkingModels/stable70beluga2-70b.ggmlv3.q5_K_S.bin -o stable70beluga2-70b.ggmlv3.q5_K_S.gguf --gqa 8 --eps 1e-5 -c 4096
#AttributeError: module 'chromadb' has no attribute 'PersistentClient' -> pip install chromadb --upgrade  
 
import sys
import os
import gc
import tkinter as tk
from tkinter import scrolledtext
from llama_cpp import Llama
import time
import platform
from tkinter import Tk
from tkinter import filedialog, Scale
from tkinter import messagebox
from tkinter import Menu
from tkinter import END
import tkinter.font as font 

print("Platform.machine: ", platform.machine())
print ("Platform.system: ", platform.system())

# Specify the path to the WorkingModels directory
directory_unix = "/data/GGUFModels"
directory_mac = "/Users/jonathanrothberg/GGUF_Models"
directory_pi = "/home/pi/GGUFModels"

initialpersistdir_unix = "/data"
initialpersistdir_mac = "/Users/jonathanrothberg"
initialpersistdir_pi = "/home/pi"

embedding_unix = "/data/all-MiniLM-L6-v2"
embedding_mac = "/Users/jonathanrothberg/all-MiniLM-L6-v2"
embedding_pi = "/home/pi/all-MiniLM-L6-v2"

# Check the system platform
if platform.system() == 'Darwin':  # Darwin stands for macOS
    directory = directory_mac
    initialpersistdir = initialpersistdir_mac
    embeddings_model_name = embedding_mac

elif platform.machine() == 'aarch64': #Pi
    directory = directory_pi
    initialpersistdir = initialpersistdir_pi
    import pysqlite3                                                                                                                                              
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')  # Needed for Pi which had old version of sqlite3
    embeddings_model_name = embedding_pi #added this locally so not in .cache
else:
    directory = directory_unix
    initialpersistdir = initialpersistdir_unix
    embeddings_model_name = embedding_unix

import chromadb # import after sqlite3 update or you get error!
from chromadb.utils import embedding_functions

print(f'Model Director: {directory}  Initial Location for Vector Database: {initialpersistdir}')

def clear_text(widget):
    if widget.winfo_class() == 'Text':
        widget.delete('1.0', END)
    elif widget.winfo_class() == 'Entry':
        widget.delete(0, END)


def show_context_menu(event):
    context_menu = Menu(root, tearoff=0)
    context_menu.configure(font=default_font)  
    context_menu.add_command(label="Cut", command=lambda: root.focus_get().event_generate("<<Cut>>"))
    context_menu.add_command(label="Copy", command=lambda: root.focus_get().event_generate("<<Copy>>"))
    context_menu.add_command(label="Paste", command=lambda: root.focus_get().event_generate("<<Paste>>"))
    context_menu.add_command(label="Clear", command=lambda: clear_text(root.focus_get()))
    context_menu.tk_popup(event.x_root, event.y_root)

# Function to change the model
def change_model(model_number):
    global model, root, model_name
    model_name = bin_files[int(model_number) - 1]
    model_path = os.path.join(directory, model_name)
    
    if 'model' in globals():
    # Delete the existing model
        print("deleting model")
        del model
        # Explicitly clear the GPU memory
        # torch.cuda.empty_cache()
        # Collect any residual garbage
        gc.collect()

    temp = tempLLM.get()
    model = Llama(model_path,n_threads=16, n_gpu_layers=100, n_ctx=4096)
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window
    #Prompt template for ChatML
    systemprompt = '''
<|im_start|> <|im_end|>
Code Llama end in <step>
'''
    
    entry.insert(tk.END, f"{systemprompt}")
# Create buttons for each available model
def create_model_buttons():
    for i, file in enumerate(bin_files, 1):
        button_text = file[:12]  # Limit the button text to the first 5 letters of the model name
        button = tk.Button(root, text=button_text, command=lambda i=i: change_model(i))
        button.grid(row=1, column=i +4, sticky='w')
    return (i) # number of buttons so we can space correctly

# Select persist directory
def select_directories():
    print("Select the Vectorstore/Persist directory for the vector search:")
    
    persist_directory = filedialog.askdirectory(initialdir= initialpersistdir)
    
    print("VectorStore/Persist directory:", persist_directory)
    return persist_directory


def set_up_chromastore():
    global collection, db, vector_folder_name
    print ("setting up db & collection")
    
    persist_directory = select_directories()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embeddings_model_name)
    
    vector_folder_name = os.path.basename(persist_directory)
    print ("path: ", persist_directory)
    
    db = chromadb.PersistentClient(path = persist_directory)
    
    collections = db.list_collections() # Debugging to get name in collectons
    print (collections)
    for collection in collections:
        print(collection.name)
    mycollection = (collections[0].name)
    print ("Collection: ", mycollection)
    #collection = db.get_collection(mycollection)
    collection = db.get_collection(mycollection, embedding_function=sentence_transformer_ef)
    
# Perform a search in the loaded database
def vector_search(query):
    docs = collection.query(query_texts =[query], n_results=3)
    vector_text = ""
    sources = ""
    #print(docs)
    for i in range(len(docs['documents'][0])):
        print("Document:", docs['documents'][0][i])
        vector_text = vector_text + "\n Vector Response "+ str(i) +": " + docs['documents'][0][i]
        source_i = docs['metadatas'][0][i]['source']
        
        if '/' in source_i:
            source_i = source_i.split('/')[-2:] #removing path from source.
        else:
            source_i = source_i
    
        source_i = ' '.join(source_i)  # convert list to string
        print ("Source: ",source_i)
        sources = sources + "\n Sources: " + source_i
        print("\n") # Just to create a new line between entries
    return vector_text, sources


def talk_to_vector_search():
    if not db:
        set_up_chromastore()
    prompt = entry.get("1.0", "end-1c") # get text from Text widget    
    vector_answers, sources = vector_search(prompt)
    #response_with_vector = model("### USER: " + prompt + "Answer based on this information: " + vector_answer)
    text_area.insert("end", f"\n\n----- {vector_folder_name} Database Search -----\n")
    text_area.see(tk.END) 
    text_area.insert(tk.END, f"{vector_answers}\n")
    
    text_area.insert(tk.END, f"{sources}\n")
    text_area.see(tk.END)  # Make the last line visible
    text_area.update() 


def save_text():
    prompt = entry.get("1.0", "end-1c")  # Get text from Text widget
    generated_text = text_area.get("1.0", "end-1c")  # Get text from Text widget
    # Create the filename using the first 10 characters of the prompt and a 4-digit timestamp
    filename = prompt[:12] + "_" + model_name[:5] + "_" + time.strftime("%m%d-%H%M") + ".txt"
    # Create a directory to save the files if it doesn't exist
    directory = "saved_texts"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the prompt and generated text to the file
    with open(os.path.join(directory, filename), "w") as file:
        file.write("Prompt:\n" + prompt + "\n\nGenerated Text:\n" + generated_text)
    print("Text saved successfully in: ",filename)


def talk_to_LLM():
    if not model_name:
        print ("No Model Loaded")
        return
   
    temp = tempLLM.get()
    tokens = max_new_tokens.get()
    root.title(("JMR's Little " + model_name + " Chat. Temp: " + str(temp))) # Set the title for the window
    prompt = entry.get("1.0", "end-1c") # get text from Text widget
  
    info = "### User: \n" + prompt + " \n### Assistant: " + model_name + " Chat. Temp: " + str(temp) + " Max Token" + str(tokens) +  "\n"
    text_area.insert(tk.END, f"{info}")
    response = model (prompt, max_tokens = tokens, stream = True, stop = ["</s>"], echo =False)
   
    #response = model (prompt, max_tokens = tokens ,temperature = temp, repeat_penalty =1.5, stream = True, stop = ["</s>"], echo =False)
    print (response) # a function when streaming.
    #text_area.insert(tk.END, f"Result:\n")
    for stream in response:
        #text = stream["choices"][0]["delta"].get("content", "")
        text = stream['choices'][0]['text']
        text_area.insert(tk.END, text)
        text_area.see(tk.END)  # Make the last line visible
        text_area.update()
    text_area.insert(tk.END, "\n\n")
    
db = False

root = tk.Tk()
new_width = 850
new_height = 800
root.geometry(f"{new_width}x{new_height}")
default_font = tk.font.nametofont("TkDefaultFont")                           
default_font.configure(size=12)  
root.title(("JMR's Little Llama.cpp GGUF Chat")) # Set the title for the window

# Check if the default path with the model exists
if not os.path.exists(directory):
    messagebox.showinfo("Information","Select Directory for GGUF llama.cpp type models")
    directory = filedialog.askdirectory(title="Select Directory for GGUF LlamaCPP type models")

# Get a list of all the .bin files in the directory
bin_files = [file for file in os.listdir(directory) if file.endswith(".bin") or file.endswith(".gguf")]

model_name = ""

# Print the numbered list of .bin files
print("Available models:")
for i, file in enumerate(bin_files, 1):
    print(f"{i}. {file}")

root.bind("<Button-2>", show_context_menu) #for macos
root.bind("<Button-3>", show_context_menu) #for linux
root.grid_rowconfigure(0, weight=1)  # Entry field takes 1 part
root.grid_rowconfigure(1, weight=0)  # "Send" button takes no extra space
root.grid_rowconfigure(2, weight=0)  # Model buttons take no extra space
root.grid_rowconfigure(3, weight=7)  # Output field takes 5 parts
root.grid_columnconfigure(0, weight=1)  # Column 0 takes all available horizontal space

# Add the create_model_buttons function call
number_of_models = create_model_buttons()  # Create buttons for each available model
#print (number_of_models)

if number_of_models > 3:
    new_width += 260 * (number_of_models - 3)
    root.geometry(f"{new_width}x{new_height}")

entry = scrolledtext.ScrolledText(root, height=5,wrap="word") # change Entry to ScrolledText, set height
entry.grid(row=0, column=0, columnspan=number_of_models+8, sticky='nsew') # make it as wide as the root window and expand with window resize
entry.configure(font = default_font)
button_V = tk.Button(root, text="Search", command=talk_to_vector_search)
button_V.grid(row=1, column=number_of_models+6, sticky='e')

button = tk.Button(root, text="Prompt", command=talk_to_LLM)
button.grid(row=1, column=number_of_models+7, sticky='e')  # place Send button in row 1, column 1, align to right

save_button = tk.Button(root, text="Save", command=save_text)
save_button.grid(row=1, column=0, sticky='w')  # Place the Save button in row 1, column 0, align to left

temp_label = tk.Label(root, text="Temperature:")
temp_label.grid(row=1, column=1, sticky='w')

tempLLM = tk.DoubleVar(value = 0.0)
slider = tk.Scale(root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=tempLLM)
slider.grid(row=1, column=2, sticky='w')
temp = tempLLM.get()

max_label = tk.Label(root, text="Max New Tokens:")
max_label.grid(row=1, column=3, sticky='w')

max_new_tokens = tk.DoubleVar(value = 256)
slider_token = tk.Scale(root, from_=0, to=4000, resolution=1, orient=tk.HORIZONTAL, variable=max_new_tokens)
slider_token.grid(row=1, column=4, sticky='w')

text_area = scrolledtext.ScrolledText(root, wrap="word", font = default_font)
text_area.grid(row=3, column=0, columnspan=number_of_models+8, sticky='nsew')  # make text area fill the rest of the window and expand with window resize, span 3 columns

root.mainloop()