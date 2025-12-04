# AI-Powered Testing Bot Benchmarking Framework

Creates a benchmark framework for AI testing bots (ChatGPT, Claude, Gemini) across three categories:
- **Test Creation** - Generate tests from source code
- **Test Execution** - Run tests and measure false positives/negatives through mutation testing
- **Test Maintenance** - Fix broken tests (if any)

All bots get **identical prompts** for fair comparison.

## Setup

```bash
# Install dependencies
pip install openai anthropic google-generativeai pytest pytest-cov --break-system-packages

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

## Output

The framework generates CSV files:
- `benchmark_summary.csv` - Overall scores for all bots
- `creation_metrics.csv` - Test creation details
- `execution_metrics.csv` - Test execution details  
- `maintenance_metrics.csv` - Test maintenance details

## Files

```
ai-testing-benchmark/
├── run_benchmark.py      # Main entry point
├── llm_bots.py           # ChatGPT, Claude, Gemini implementations
├── standard_prompts.py   # Standardized prompts (same for all bots)
├── measure_creation.py   # Test creation measurement
├── measure_execution.py  # Test execution measurement
├── measure_maintenance.py # Test maintenance measurement
├── generate_report.py    # CSV report generation
└── benchmark_framework.py # Core data classes
```
