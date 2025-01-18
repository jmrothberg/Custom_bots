# gguf_model_tester.py
# Jan 11 2025
# Jonathan Rothberg
# This is a simple gradio interface to test llama.cpp models in GGUF format.
# It allows you to select a model, enter a system prompt, and generate responses.
# It also allows you to enter a prompt and see the model's response.
# It uses the llama.cpp library to load and generate responses from the models.
# It uses the gradio library to create the interface.
# It uses the os library to get the list of models and to join paths.
# It uses the glob library to get the list of models.
# It uses the gc library to force garbage collection when a new model is loaded.
# had to change the code to use the namespace for the execution of the code to use classes and pygames
# access claude from environment variable
import gradio as gr
import os
from llama_cpp import Llama
import glob
import gc
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict
import time
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from loguru import logger
import better_exceptions
import sys
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from io import StringIO
import psutil  # You may need to: pip install psutil
import signal
import anthropic

MODELS_DIR = "/data/GGUF_Models"
CODE_DIR = "LLM_generated_code"  # This is correct

# Make sure the code directory exists
if not os.path.exists(CODE_DIR):
    os.makedirs(CODE_DIR)

running_process = None
current_model = None
current_model_name = None

# 1) Add a global flag for stopping generation at the top:
stop_generation_flag = False

# Load Anthropic API key from environment variable
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY
)

def get_available_models():
    """Get list of .gguf files in the models directory"""
    models = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
    return [os.path.basename(m) for m in models]

def load_model(model_name):
    """Load the selected GGUF model"""
    global current_model, current_model_name
    if not model_name:
        return None
    
    if current_model_name != model_name:
        # Clean up old model if it exists
        if current_model is not None:
            del current_model
            
            gc.collect()  # Force garbage collection
        
        model_path = os.path.join(MODELS_DIR, model_name)
        current_model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=16000  # Set larger context window at model initialization
        )
        current_model_name = model_name
    return current_model

def generate_response(model_name, system_prompt, prompt, max_tokens=4096, temperature=0.1, use_claude=False):
    """Generate response from the model"""
    global current_model, current_model_name, stop_generation_flag
    stop_generation_flag = False  # Reset the flag at start
    
    if use_claude:  # Add Claude handling
        print("Using Claude 3 Sonnet")
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        yield response.content[0].text
        return

    print(f"Generating response with model {model_name}...")
    print (f"max_tokens: {max_tokens}, temperature: {temperature}")
    if not model_name:
        return "Please select a model first."
    
    # Load model if needed
    if current_model_name != model_name:
        current_model = load_model(model_name)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    response_text = ""
    # The yield works with Gradio's streaming by:
    # 1. Yielding each update to the text
    # 2. Gradio automatically updates the output textbox with each yield
    for output in current_model.create_chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        stream=True
    ):
        if stop_generation_flag:
            break
        delta = output['choices'][0]['delta'].get('content', '')
        response_text += delta
        yield response_text  # Each yield updates the Gradio interface

# Add these new classes after your imports
@dataclass
class CodeAttempt:
    """Track each code execution attempt"""
    timestamp: str
    code: str
    output: str
    errors: str
    success: bool
    execution_time: float

class DebugTracker:
    def __init__(self):
        self.attempts = []
    
    def add_attempt(self, attempt: CodeAttempt):
        self.attempts.append(attempt)
    
    def get_history(self) -> str:
        if not self.attempts:
            return "No previous attempts"
        
        history = "=== Execution History ===\n"
        for i, attempt in enumerate(self.attempts[-5:], 1):  # Show last 5 attempts
            history += f"\nAttempt {i}:\n"
            history += f"Time: {attempt.timestamp}\n"
            history += f"Success: {attempt.success}\n"
            history += f"Duration: {attempt.execution_time:.2f}s\n"
            if not attempt.success:
                history += f"Errors: {attempt.errors}\n"
            history += "---\n"
        return history

# Add these global variables after your existing ones
debug_tracker = DebugTracker()

# Add this enhanced system prompt
ENHANCED_SYSTEM_PROMPT = '''You are a Python debugging assistant. For all code:
1. Add comprehensive debug prints showing:
   - Function entry/exit points with timestamps
   - All variable states and changes
   - Loop iterations and conditions
   - Event triggers and state changes
   - Performance metrics and timing
2. Include detailed try-except blocks with:
   - Specific exception types
   - Detailed error messages
   - State information at point of failure
3. Add performance tracking:
   - Function execution times
   - Resource usage where relevant
   - Operation counts in loops
4. For PyGame/GUI apps, add:
   - Event logging
   - State change tracking
   - FPS and timing metrics
   - Input/output validation


Use print statements liberally for debugging:
- f"[DEBUG][{{function_name}}] Starting with params: {{params}}"
- f"[INFO][{{component}}] Variable state: {{var_name}}={{var_value}}"
- f"[PERF][{{section}}] Execution time: {{time_taken:.3f}}s"

Previous errors and execution history will be provided.
Analyze them carefully and improve the code while maintaining functionality.
Focus on making the code more debuggable and self-documenting.
'''

# Add these utility functions
def get_timestamp():
    """Generate timestamp for logging"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def debug_print(level: str, component: str, message: str):
    """Formatted debug printing"""
    print(f"[{get_timestamp()}][{level}][{component}] {message}")

# Configure better exceptions
better_exceptions.MAX_LENGTH = 100  # Keep tracebacks readable
better_exceptions.hook()

class CodeDebugger:
    """Robust debugger using loguru"""
    def __init__(self):
        self.log_capture = StringIO()
        self.start_time = None
        
        # Configure loguru for our capture
        logger.remove()  # Remove default handler
        logger.add(self.log_capture, format="{time} | {level} | {message}", level="DEBUG")
    
    @contextmanager
    def capture(self):
        """Context manager for capturing output and errors"""
        self.start_time = time.time()
        try:
            yield self
        finally:
            logger.remove()  # Cleanup

    def get_summary(self):
        return self.log_capture.getvalue()

def safe_execute_code(code_str: str):
    """Execute code - in console if no input needed, in terminal if input is present"""
    print("[DEBUG] Entering safe_execute_code")
    try:
        # Check if code contains any input() statements
        if 'input(' in code_str and 'pygame' not in code_str:  # Don't use terminal for pygame
            print("[DEBUG] Detected input() - about to run xterm")  # Debug print
            # Add print before first input if there isn't one
            if 'print' not in code_str.split('input(')[0]:
                code_str = 'print("Program starting...")\n' + code_str
            
            # Add pause at end
            code_str = code_str + '\nprint("\\nPress Enter to exit...")\ninput()'
            
            # Escape single quotes in the Python code and wrap in single quotes
            escaped_code = code_str.replace("'", "'\\''")
            # Use xterm for a smaller, white terminal window
            os.system(f"xterm -bg white -fg black -geometry 80x24 -e python3 -c '{escaped_code}'")
            
            return {  # This return statement MUST be here to prevent double execution
                'success': True,
                'output': "Code ran in terminal window",
                'execution_time': 0,
                'errors': None
            }
        else:
            # Run in execution console if no input needed or if it's pygame
            output_buffer = StringIO()
            start_time = time.time()
            
            try:
                with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                    # Create a new namespace for execution
                    namespace = {}
                    # Add any required built-ins or globals
                    namespace.update({
                        'print': print,
                        '__builtins__': __builtins__,
                    })
                    
                    # Conditionally import pygame if needed
                    if 'pygame' in code_str:
                        import pygame
                        namespace['pygame'] = pygame
                    
                    try:
                        # Execute in the dedicated namespace
                        exec(code_str, namespace)
                    except SystemExit:
                        return {
                            'success': True,
                            'output': output_buffer.getvalue() + "\nProgram exited successfully",
                            'execution_time': time.time() - start_time,
                            'errors': None
                        }
                success = True
                errors = None
            except Exception as e:
                success = False
                errors = f"{str(e)}\n{traceback.format_exc()}"
            
            execution_time = time.time() - start_time
            output = output_buffer.getvalue()
            
            return {
                'success': success,
                'output': output,
                'execution_time': execution_time,
                'errors': errors
            }
            
    except Exception as e:
        error_info = f"{str(e)}\n{traceback.format_exc()}"
        return {
            'success': False,
            'output': str(e),
            'execution_time': 0,
            'errors': error_info
        }

# Add this function to generate prompts for the LLM
def generate_fix_prompt(code: str, result: dict, history: str, refinement_text: str = "") -> str:
    prompt = f"""
Code needs improvement or debugging.

Original Code:
```python
{code}
```

Execution Results:
- Success: {result['success']}
- Time: {result['execution_time']:.2f}s
- Output:
```
{result['output']}
```

Debug History:
```
{history}
```
"""

    if result['errors']:
        prompt += f"""
Errors:
```
{result['errors']}
{result['traceback'] if result['traceback'] else ''}
```
"""

    if refinement_text:
        prompt += f"""
Requested Refinements:
{refinement_text}
"""

    prompt += """
Please:
1. Add extensive debug prints for:
   - All function entry/exit points
   - Variable states and changes
   - Conditional logic results
   - Performance metrics
2. Fix any identified issues
3. Apply requested refinements
4. Enhance error handling with detailed messages
5. Explain your changes and debug strategy

Make the code as self-documenting and debuggable as possible.
"""
    return prompt

# Add this function near your other utility functions
def extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks"""
    if "```python" in text and "```" in text:
        try:
            code = text.split("```python")[1].split("```")[0].strip()
            return code
        except IndexError:
            return text
    return text

def save_code_to_file(code: str, filename: str = None) -> str:
    """Save code to a file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_code_{timestamp}.py"
    
    filepath = os.path.join(CODE_DIR, filename)
    try:
        with open(filepath, 'w') as f:
            f.write(code)
        return f"Code saved successfully to {filepath}"
    except Exception as e:
        return f"Error saving code: {str(e)}"

def get_saved_code_files():
    """Get list of saved Python files"""
    files = glob.glob(os.path.join("LLM_generated_code", "*.py"))  # Match the actual directory
    return [os.path.basename(f) for f in files]

def load_code_from_file(filename: str) -> tuple[str, str]:
    """Load code from a file"""
    try:
        filepath = os.path.join(CODE_DIR, filename)
        with open(filepath, 'r') as f:
            code = f.read()
        return code, f"Code loaded successfully from {filepath}"
    except Exception as e:
        return "", f"Error loading code: {str(e)}"

# Add both these functions before the Gradio interface section (before `with gr.Blocks() as demo:`)
def stop_generation():
    """Stop the LLM generation"""
    global stop_generation_flag
    stop_generation_flag = True
    return [
        gr.update(value="Generation stopped", interactive=True),
        gr.update(value=extract_python_code("Generation stopped"))
    ]

def stop_execution():
    """Stop the running code (e.g. if pygame is running)."""
    try:
        import pygame
        pygame.display.quit()
        pygame.quit()
    except:
        pass
    return "Code execution stopped"


# Add these new handler functions
def execute_and_track(code_text):
    """Execute code and track results"""
    if not code_text:
        return "No code to execute", "No execution history"
    
    code = code_text.strip()
    if not code:
        return "No valid Python code found", "No execution history"
    
    result = safe_execute_code(code)
    
    # Add attempt to debug tracker with full debug information
    debug_tracker.add_attempt(CodeAttempt(
        timestamp=get_timestamp(),
        code=code,
        output=result['output'],
        errors=result['errors'],
        success=result['success'],
        execution_time=result['execution_time']
    ))
    
    # Simplified debug history without accessing non-existent keys
    debug_history = debug_tracker.get_history() + "\n"
    debug_history += f"=== Latest Execution Details ===\n"
    debug_history += f"Output: {result['output']}\n"
    if result['errors']:
        debug_history += f"Errors: {result['errors']}\n"
    
    return result['output'], debug_history

def generate_fix_response(model_name, system_prompt, extracted_code, execution_output, debug_history, input_text, max_tokens=4096, temperature=0.1, use_claude=False):
    """Generate fix response from the model, similar to generate_response but with additional context"""
    global current_model, current_model_name, stop_generation_flag
    stop_generation_flag = False  # Reset the flag at start
    
    # Use the enhanced system prompt instead of the original
    system_prompt = ENHANCED_SYSTEM_PROMPT
    
    context_prompt = f"""
Previous Code:
```python
{extracted_code}
```

Execution Output:
{execution_output}

Debug History:
{debug_history}

Additional Instructions:
{input_text}
"""
    
    if use_claude:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": context_prompt}],
        )
        yield response.content[0].text
        return

    print(f"Generating fix with model {model_name}...")
    print(f"max_tokens: {max_tokens}, temperature: {temperature}")
    
    if not model_name:
        return "Please select a model first."
    
    # Load model if needed
    if current_model_name != model_name:
        current_model = load_model(model_name)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_prompt}
    ]
    
    response_text = ""
    for output in current_model.create_chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        stream=True
    ):
        if stop_generation_flag:
            break
        delta = output['choices'][0]['delta'].get('content', '')
        response_text += delta
        yield response_text

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# GGUF Model Tester")
    
    with gr.Row():
        with gr.Column(scale=2):  # Left column for model selection and Claude
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    value=None,
                    allow_custom_value=False,
                    scale=2  # Takes up 2/3 of the left column
                )
                use_claude = gr.Checkbox(
                    label="Use Claude 3 Sonnet",
                    value=False,
                    scale=1  # Takes up 1/3 of the left column
                )
        
        with gr.Column(scale=1):  # Right column for refresh button
            refresh_btn = gr.Button("🔄 Refresh Models")
    
    with gr.Row():
        system_prompt = gr.Textbox(
            lines=2,
            label="System Prompt",
            value="""You are a Python coding assistant. Please respond with working Python code examples.
            Always wrap your code in ```python and ``` markers. Include helpful comments and debug prints.
            Focus on providing clean, executable code that demonstrates the solution.""",
            placeholder="Enter system prompt here..."
        )
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=5,
                label="Input Text",
                placeholder="Enter your prompt here..."
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=4096,
                    step=1,
                    label="Max Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.1,
                    step=0.1,
                    label="Temperature"
                )
            with gr.Row():  # New row for generate buttons
                submit_btn = gr.Button("Generate")
                stop_gen_btn = gr.Button("⏹️ Stop", scale=1)  # Fixed scale to integer
        
        output_text = gr.Textbox(
            lines=10,
            label="Model Output"
        )
    
    with gr.Row():
        extracted_code = gr.Code(
            label="Extracted Python Code",
            language="python",
            interactive=True
        )
    
    with gr.Row():
        run_code_btn = gr.Button("Run Code")
        stop_run_btn = gr.Button("⏹️ Stop", scale=1)  # Fixed scale to integer
        fix_code_btn = gr.Button("Fix Code")
        save_code_btn = gr.Button("Save Code")
        load_code_btn = gr.Button("Load Code")
        
    with gr.Row():
        filename_input = gr.Textbox(
            label="Save Filename",
            placeholder="Enter filename to save (optional)"
        )
        load_file_dropdown = gr.Dropdown(
            choices=get_saved_code_files(),
            label="Select Saved Code",
            interactive=True
        )
        refresh_files_btn = gr.Button("🔄 Refresh Files")
    
    with gr.Row():
        execution_output = gr.Textbox(
            lines=10,
            label="Execution Console",
            interactive=True
        )
        history_view = gr.Textbox(
            lines=10,
            label="Debug History",
            interactive=True
        )
    
    # Handle refresh button
    def refresh_models():
        return gr.Dropdown(choices=get_available_models())
    refresh_btn.click(refresh_models, outputs=[model_dropdown])
    
    # Wire up the generation with cancellation
    submit_event = submit_btn.click(
        generate_response,
        inputs=[model_dropdown, system_prompt, input_text, max_tokens, temperature, use_claude],
        outputs=output_text,
        api_name="generate"
    )
    
    # Wire up the run code with cancellation
    run_event = run_code_btn.click(
        #lambda x: safe_execute_code(x),  # Use the existing function
        execute_and_track,
        inputs=[extracted_code],
        outputs=[execution_output, history_view]
    )
    
    
    
    def preview_extracted_code(text):
        """Preview the extracted code"""
        code = extract_python_code(text)
        return code
    
    def handle_save_code(code, filename):
        """Handle saving the code"""
        if not code:
            return "No code to save"
        return save_code_to_file(code, filename if filename.strip() else None)
    
    def handle_load_code(filename):
        """Load selected code file"""
        if not filename:
            return "", "No file selected"
        try:
            with open(os.path.join("LLM_generated_code", filename), 'r') as f:  # Match the actual directory
                code = f.read()
            return code, f"Loaded {filename}"
        except Exception as e:
            return "", f"Error loading file: {str(e)}"
    
    # Wire up the new handlers
    output_text.change(
        preview_extracted_code,
        inputs=[output_text],
        outputs=[extracted_code]
    )
    
    save_code_btn.click(
        handle_save_code,
        inputs=[extracted_code, filename_input],
        outputs=[execution_output]
    )
    
    
    fix_code_btn.click(
        generate_fix_response,
        inputs=[
            model_dropdown,
            system_prompt,
            extracted_code,
            execution_output,
            history_view,
            input_text,
            max_tokens,
            temperature,
            use_claude
        ],
        outputs=output_text
    )
    
    load_code_btn.click(
        handle_load_code,
        inputs=[load_file_dropdown],  # Changed from filename_input to load_file_dropdown
        outputs=[extracted_code, execution_output]
    )

    # Add refresh handler
    def refresh_code_files():
        return gr.Dropdown(choices=get_saved_code_files())

    # Wire up the handlers
    refresh_files_btn.click(
        refresh_code_files,
        outputs=[load_file_dropdown]
    )

    # Add stop handlers correctly
    stop_gen_btn.click(
        stop_generation,
        outputs=[output_text, extracted_code],
        cancels=[submit_event]
    )
    stop_run_btn.click(
        stop_execution,
        outputs=[execution_output],
        cancels=[run_event]  # Keep this to cancel the run event
    )

if __name__ == "__main__":
    demo.launch(share=False) 