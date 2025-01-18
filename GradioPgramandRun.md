# GGUF Model Tester

A Gradio-based interface for testing and debugging LLaMA.cpp models in GGUF format, with support for Claude 3 Sonnet integration.

## Environment Setup

Before running the application, set your Anthropic API key as an environment variable:

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-api-key-here"

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=your-api-key-here

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="your-api-key-here"
```
You can also create a `.env` file in the project root and use python-dotenv to load it:

```env
ANTHROPIC_API_KEY=your-api-key-here
```
## Features

- 🤖 Test multiple GGUF format models
- 🔄 Real-time code generation and execution
- 🐛 Comprehensive debugging and error tracking
- 💾 Save and load generated code
- 🎮 Support for PyGame applications
- 🔍 Detailed execution history
- 🛑 Generation and execution stopping capabilities
- 🤖 Optional Claude 3 Sonnet integration

## Prerequisites

```bash
pip install gradio llama-cpp-python anthropic loguru better_exceptions psutil pygame
```

## Configuration

1. Set up your models directory:
```python
MODELS_DIR = "/data/GGUF_Models"  # Update this path to your GGUF models location
```

2. (Optional) Configure Claude API:
```python
# Add your Claude API key in the code
client = anthropic.Anthropic(api_key="your-api-key-here")
```

## Usage

1. Run the application:
```bash
python GradioProgramandRun.py
```

2. Access the interface through your web browser (typically at `http://localhost:7860`)

3. Features available in the interface:
   - Select GGUF models from dropdown
   - Toggle Claude 3 Sonnet integration
   - Set system prompts
   - Generate and execute code
   - Save/load generated code
   - Track execution history and debug information

## Code Generation Features

- Comprehensive debug prints
- Function entry/exit tracking
- Variable state monitoring
- Performance metrics
- Detailed error handling
- PyGame/GUI event logging

## File Management

Generated code is saved in the `LLM_generated_code` directory with features to:
- Save code with custom filenames
- Load previously saved code
- Refresh file listings
- Track execution history

## Debug Features

- Timestamp logging
- Execution time tracking
- Resource usage monitoring
- Detailed error tracebacks
- Historical attempt tracking

## Safety Features

- Isolated code execution
- Terminal handling for input-required code
- Process monitoring and termination
- Error capture and reporting

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Your chosen license]

## Author

Jonathan Rothberg

## Acknowledgments

- LLaMA.cpp project
- Gradio team
- Anthropic's Claude
