# AI-Powered Testing Bot Benchmarking Framework

A benchmark framework for AI testing bots (ChatGPT, Claude, Gemini) across three categories:
- **Test Creation** - Generate tests from source code and measure coverage
- **Test Execution** - Run tests and measure false positives/negatives through mutation testing
- **Test Maintenance** - Fix broken tests (if any)

All bots get **identical prompts** for fair comparison.

## Setup

```bash
# Install dependencies
pip install python-dotenv openai anthropic google-generativeai pytest pytest-cov --break-system-packages

# Set API keys in an .env file:
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
```

Add your source code into the respective folders as follows:
<language>\<projectName>\src\HERE

For example, if I wanted to add a Python project called Slugify, I would add it to the python folder, name the directory "slugify", and then put the source code slugify.py into the src folder.

## Usage

```bash
# Benchmark all avaiable projects for a specific language with all bots 
python run_benchmark.py --all --language python

# Benchmark a specific bot
python run_benchmark.py --bot chatgpt --language python --project ./src

# Benchmark all available bots
python run_benchmark.py --all --language python --project ./src

# Specify output directory
python run_benchmark.py --all --language python --project ./src --output ./results

```

## Command Line Options
| Option     | Description                                                                |
|------------|----------------------------------------------------------------------------|
| --bot      | Bot to use: chatgpt, claude, gemini, or all                                |
| --language | Language: python, java, javascript, or all                                 |
| --project  | Path to source code directory (optional - auto-discovers if not specified) |
| --output   | Output directory (default: ./benchmark_results)                            |
| --debug    | Enable debug output for troubleshooting                                    |

## Output

The framework generates CSV files:
- `benchmark_summary.csv` - Overall scores for all bots
- `creation_metrics.csv` - Test creation details
- `execution_metrics.csv` - Test execution details  
- `maintenance_metrics.csv` - Test maintenance details

## Project Structure
Projects should be organized as follows:

```
python/
├── project1/
│   └── src/
│       ├── module1.py
│       └── module2.py
├── project2/
│   └── src/
│       └── code.py
```
Projects are auto-discovered when --project is not specified.