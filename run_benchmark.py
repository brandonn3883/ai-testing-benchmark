#!/usr/bin/env python3
"""
AI Testing Bot Benchmark Runner

Benchmarks ChatGPT, Claude, and Gemini for test generation, execution, and maintenance.

Per-file flow:
1. Generate test for 1 file
2. Run coverage
3. If error, fix that file (maintenance)
4. If no error, run mutation testing (execution)
5. Write metrics for that file
6. Move to next file

Usage:
    python run_benchmark.py --bot chatgpt --language python --project ./src
    python run_benchmark.py --bot all --language python --project ./src
"""

import argparse
import os
import re
import json
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Optional

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_bots import create_bot, get_available_bots, BotInterface


@dataclass
class FileResult:
    """Results for a single source file through the entire pipeline."""
    # Identification
    bot_name: str
    language: str
    project_name: str
    file_name: str
    timestamp: str = ""
    
    # Creation metrics
    test_generated: bool = False
    generation_time_seconds: float = 0.0
    line_coverage_pct: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    
    # Maintenance metrics
    had_errors: bool = False
    errors_fixed: bool = False
    errors_remain: bool = False
    fix_attempts: int = 0
    first_attempt_fix: bool = False
    
    # Execution metrics (mutation testing)
    mutation_score_pct: float = 0.0
    mutants_total: int = 0
    mutants_killed: int = 0
    mutants_survived: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BenchmarkRunner:
    """Runs per-file benchmarks on AI testing bots."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.output_dir / "work"
        self.work_dir.mkdir(exist_ok=True)
        self.results: List[FileResult] = []
    
    def run_benchmark(
        self,
        bot: BotInterface,
        project_path: str,
        language: str,
        project_name: str = None,
        verbose: bool = True,
        debug: bool = False
    ) -> List[FileResult]:
        """
        Run per-file benchmark on a project.
        
        For each source file:
        1. Generate test
        2. Run coverage
        3. If error, fix (maintenance)
        4. If no error, run mutation testing (execution)
        5. Record all metrics
        """
        project_path = Path(project_path).resolve()
        
        if project_name is None:
            project_name = project_path.name
            if project_name.lower() in ("src", "lib", "source", "main", "app"):
                project_name = project_path.parent.name
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING: {bot.name}")
            print(f"Project: {project_path}")
            print(f"Language: {language}")
            print(f"{'='*60}")
        
        if language == "python":
            return self._run_python_benchmark(
                bot, project_path, project_name, verbose, debug
            )
        elif language == "java":
            return self._run_java_benchmark(
                bot, project_path, project_name, verbose, debug
            )
        elif language == "javascript":
            return self._run_javascript_benchmark(
                bot, project_path, project_name, verbose, debug
            )
        else:
            print(f"Unsupported language: {language}")
            return []
    
    def _run_python_benchmark(
        self,
        bot: BotInterface,
        project_path: Path,
        project_name: str,
        verbose: bool,
        debug: bool
    ) -> List[FileResult]:
        """Run per-file benchmark for Python."""
        
        # Find all source files
        all_source_files = list(project_path.rglob("*.py"))
        all_source_files = [f for f in all_source_files if not f.name.startswith("test_")]
        
        # Files to test (exclude __init__.py and _private.py)
        testable_files = [f for f in all_source_files 
                         if f.name != "__init__.py" and not f.name.startswith("_")]
        
        if verbose:
            print(f"\n    Found {len(testable_files)} testable file(s)")
        
        # Create work directory
        test_dir = self.work_dir / "generated_tests"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ALL source files (with relative imports fixed)
        if debug:
            print(f"    [DEBUG] Copying source files to {test_dir}")
        for src_file in all_source_files:
            content = src_file.read_text(encoding='utf-8', errors='replace')
            
            # Fix relative imports ONLY (starting with .)
            # from .module import x -> from module import x
            content = re.sub(r'from\s+\.(\w+)\s+import', r'from \1 import', content)
            # from . import module -> import module
            content = re.sub(r'from\s+\.\s+import\s+(\w+)', r'import \1', content)
            # from ..module import x -> from module import x
            content = re.sub(r'from\s+\.\.(\w+)\s+import', r'from \1 import', content)
            # from .. import module -> import module
            content = re.sub(r'from\s+\.\.\s+import\s+(\w+)', r'import \1', content)
            # from .package.module import -> from package.module import  
            content = re.sub(r'from\s+\.(\w+\.\w+)\s+import', r'from \1 import', content)
            # from ...module import (3 dots) -> from module import
            content = re.sub(r'from\s+\.\.\.+(\w+)\s+import', r'from \1 import', content)
            
            dest_file = test_dir / src_file.name
            dest_file.write_text(content, encoding='utf-8')
            if debug:
                print(f"    [DEBUG]   Copied: {src_file.name}")
        
        if debug:
            print(f"    [DEBUG] Files in test_dir: {[f.name for f in test_dir.glob('*.py')]}")
        
        # Create __init__.py to make test_dir a package (helps with imports)
        init_file = test_dir / "__init__.py"
        init_file.write_text("", encoding='utf-8')
        
        # Create conftest.py to add test_dir to path
        conftest_file = test_dir / "conftest.py"
        conftest_content = '''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
'''
        conftest_file.write_text(conftest_content, encoding='utf-8')
        
        # Build dependency map for context
        module_contents = {}
        for sf in all_source_files:
            module_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        # Process each file
        results = []
        for idx, source_file in enumerate(testable_files):
            if verbose:
                print(f"\n    [{idx + 1}/{len(testable_files)}] {source_file.name}")
            
            result = self._process_python_file(
                bot=bot,
                source_file=source_file,
                test_dir=test_dir,
                project_name=project_name,
                module_contents=module_contents,
                all_source_files=all_source_files,
                verbose=verbose,
                debug=debug
            )
            results.append(result)
            self.results.append(result)
        
        # Print summary
        if verbose:
            self._print_project_summary(results)
        
        return results
    
    def _process_python_file(
        self,
        bot: BotInterface,
        source_file: Path,
        test_dir: Path,
        project_name: str,
        module_contents: dict,
        all_source_files: list,
        verbose: bool,
        debug: bool
    ) -> FileResult:
        """Process a single Python file through the complete pipeline."""
        
        result = FileResult(
            bot_name=bot.name,
            language="python",
            project_name=project_name,
            file_name=source_file.name
        )
        
        source_code = source_file.read_text(encoding='utf-8', errors='replace')
        module_name = source_file.stem
        test_filename = f"test_{module_name}.py"
        test_path = test_dir / test_filename
        
        # ========== STEP 1: Generate Test ==========
        if verbose:
            print(f"        [1/3] Test Creation...")
        start_time = time.time()
        
        # Build context with dependencies
        code_with_context = self._build_python_context(
            source_code, module_name, module_contents, all_source_files, source_file
        )
        
        try:
            generated_test = bot.generate_tests(
                code_with_context, 
                module_name=module_name, 
                language="python"
            )
            if generated_test:
                test_path.write_text(generated_test, encoding='utf-8')
                result.test_generated = True
                if verbose:
                    print(f"              ✓ Test generated")
            else:
                if verbose:
                    print(f"              ✗ No test generated")
                return result
        except Exception as e:
            if verbose:
                print(f"              ✗ Generation error: {e}")
            return result
        
        result.generation_time_seconds = time.time() - start_time
        
        # ========== STEP 2: Run Coverage ==========
        coverage, passed, failed, error_msg = self._run_python_coverage(
            test_path, module_name, test_dir, debug
        )
        
        result.tests_passed = passed
        result.tests_failed = failed
        
        # ========== STEP 3: If Error, Fix (Maintenance) ==========
        if error_msg:
            if verbose:
                print(f"        [2/3] Test Maintenance...")
            result.had_errors = True
            result.errors_remain = True
            if verbose:
                print(f"              ✗ Tests have errors, attempting fix...")
            
            for attempt in range(3):
                result.fix_attempts += 1
                try:
                    current_test = test_path.read_text(encoding='utf-8', errors='replace')
                    fixed_test = bot.repair_test(
                        current_test, error_msg, "test_error"
                    )
                    
                    if fixed_test:
                        test_path.write_text(fixed_test, encoding='utf-8')
                        
                        # Re-run coverage
                        coverage, passed, failed, error_msg = self._run_python_coverage(
                            test_path, module_name, test_dir, debug
                        )
                        
                        result.tests_passed = passed
                        result.tests_failed = failed
                        
                        if not error_msg:
                            result.errors_fixed = True
                            result.errors_remain = False
                            if attempt == 0:
                                result.first_attempt_fix = True
                            if verbose:
                                print(f"              ✓ Fixed on attempt {attempt + 1}")
                            break
                except Exception as e:
                    if debug:
                        print(f"              [DEBUG] Fix attempt {attempt + 1} failed: {e}")
            
            if result.errors_remain and verbose:
                print(f"              ✗ Could not fix after {result.fix_attempts} attempts")
        else:
            if verbose:
                print(f"        [2/3] Test Maintenance...")
                print(f"              No errors to fix")
        
        # Record coverage
        if coverage is not None:
            result.line_coverage_pct = coverage
            if verbose:
                print(f"              Coverage: {coverage:.1f}%")
        
        # ========== STEP 4: If No Error, Run Mutation Testing (Execution) ==========
        if verbose:
            print(f"        [3/3] Test Execution (Mutation Testing)...")
        
        if not result.errors_remain and result.tests_passed > 0:
            mutation_results = self._run_python_mutation(
                source_file, test_path, test_dir, module_name, debug
            )
            
            result.mutants_total = mutation_results.get("total", 0)
            result.mutants_killed = mutation_results.get("killed", 0)
            result.mutants_survived = mutation_results.get("survived", 0)
            
            if result.mutants_total > 0:
                result.mutation_score_pct = (result.mutants_killed / result.mutants_total) * 100
                if verbose:
                    print(f"              Mutants: {result.mutants_killed}/{result.mutants_total} killed ({result.mutation_score_pct:.1f}%)")
            else:
                if verbose:
                    print(f"              No mutants generated")
        else:
            if verbose:
                if result.errors_remain:
                    print(f"              Skipped (errors remain)")
                else:
                    print(f"              Skipped (no passing tests)")
        
        return result
    
    def _build_python_context(
        self, 
        source_code: str, 
        module_name: str, 
        module_contents: dict, 
        all_source_files: list,
        source_file: Path
    ) -> str:
        """Build source code with dependency context."""
        imports = []
        for match in re.finditer(r'(?:from\s+(\w+)\s+import|^import\s+(\w+))', source_code, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                imports.append(module)
        
        dependency_code = ""
        for imp in imports:
            if imp in module_contents and imp != module_name:
                dep_content = module_contents[imp]
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n# ... (truncated)"
                dependency_code += f"\n\n# === DEPENDENCY: {imp}.py ===\n{dep_content}"
        
        if dependency_code:
            return source_code + "\n\n# === PROJECT DEPENDENCIES ===" + dependency_code
        elif len(all_source_files) > 1:
            other_files = [sf.name for sf in all_source_files if sf != source_file]
            if other_files:
                return source_code + f"\n\n# NOTE: Project also contains: {', '.join(other_files)}"
        
        return source_code
    
    def _run_python_coverage(
        self, 
        test_path: Path, 
        module_name: str, 
        test_dir: Path, 
        debug: bool
    ) -> tuple:
        """
        Run pytest with coverage.
        Returns: (coverage_pct, tests_passed, tests_failed, error_message)
        """
        # Fix imports in test file
        test_content = test_path.read_text(encoding='utf-8', errors='replace')
        original_content = test_content
        
        # Fix import patterns that LLMs generate for LOCAL project imports
        # Only fix 'src.' prefix which is a common LLM pattern
        test_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
        
        # Add sys.path fix at the top of the test file if not already present
        if "sys.path" not in test_content:
            path_fix = "import sys\nimport os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n\n"
            test_content = path_fix + test_content
        
        if test_content != original_content:
            test_path.write_text(test_content, encoding='utf-8')
            if debug:
                print(f"              [DEBUG] Fixed imports in test file")
        
        # Verify source file exists in test_dir
        source_file = test_dir / f"{module_name}.py"
        if not source_file.exists():
            if debug:
                print(f"              [DEBUG] WARNING: Source file {module_name}.py not found in {test_dir}")
                print(f"              [DEBUG] Available files: {[f.name for f in test_dir.glob('*.py')]}")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_dir)
        
        # Clean old coverage
        coverage_json = test_dir / "coverage.json"
        if coverage_json.exists():
            coverage_json.unlink()
        
        cmd = [
            "python", "-m", "pytest",
            test_path.name,
            f"--cov={module_name}",
            "--cov-report=json:coverage.json",
            "--cov-report=term",
            "-v", "--tb=long"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                env=env,
                timeout=120
            )
            
            output = result.stdout + result.stderr
            
            if debug:
                print(f"        [DEBUG] pytest return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"        [DEBUG] Output (last 300): {output[-300:]}")
            
            # Parse test counts
            passed = 0
            failed = 0
            passed_match = re.search(r'(\d+) passed', output)
            failed_match = re.search(r'(\d+) failed', output)
            error_match = re.search(r'(\d+) error', output)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            if error_match:
                failed += int(error_match.group(1))
            
            # Check for errors (collection errors, import errors, etc.)
            error_msg = None
            if result.returncode != 0:
                # Check for actual errors vs just test failures
                if "Error" in output or "error" in output:
                    if "FAILED" not in output or passed == 0:
                        # Try to find the actual error line
                        # Look for ModuleNotFoundError or No module named
                        module_error = re.search(
                            r'(ModuleNotFoundError|No module named)[^\n]+',
                            output
                        )
                        if module_error:
                            error_msg = module_error.group(0)
                        else:
                            # Look for any Error line
                            error_pattern = re.search(
                                r'E\s+((?:Syntax|Import|Name|Type|Attribute|Module|ModuleNotFound)Error[^\n]+)', 
                                output
                            )
                            if error_pattern:
                                error_msg = error_pattern.group(1)
                            else:
                                # Fall back to last 500 chars
                                error_msg = output[-500:]
                        
                        if debug:
                            print(f"              [DEBUG] Full output (last 800): {output[-800:]}")
            
            # Parse coverage
            coverage_pct = None
            
            # Try JSON
            if coverage_json.exists():
                try:
                    with open(coverage_json, encoding='utf-8') as f:
                        cov_data = json.load(f)
                        totals = cov_data.get("totals", {})
                        coverage_pct = totals.get("percent_covered", 0.0)
                    coverage_json.unlink()
                except:
                    pass
            
            # Fallback to terminal
            if coverage_pct is None:
                cov_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
                if cov_match:
                    coverage_pct = float(cov_match.group(1))
            
            # Cleanup
            dotcoverage = test_dir / ".coverage"
            if dotcoverage.exists():
                dotcoverage.unlink()
            
            return coverage_pct or 0.0, passed, failed, error_msg
            
        except subprocess.TimeoutExpired:
            return 0.0, 0, 0, "Timeout"
        except Exception as e:
            return 0.0, 0, 0, str(e)
    
    def _run_python_mutation(
        self, 
        source_file: Path, 
        test_path: Path,
        test_dir: Path, 
        module_name: str, 
        debug: bool
    ) -> dict:
        """Run mutation testing on a single file."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        source_in_test_dir = test_dir / source_file.name
        if not source_in_test_dir.exists():
            return results
        
        original_code = source_in_test_dir.read_text(encoding='utf-8', errors='replace')
        
        # Generate mutants
        mutants = self._generate_python_mutants(original_code)
        
        if debug:
            print(f"        [DEBUG] Generated {len(mutants)} mutants")
        
        # Test each mutant (limit to 20)
        for mutant_code in mutants[:20]:
            killed = self._test_python_mutant(
                source_in_test_dir, mutant_code, original_code,
                test_path, test_dir
            )
            if killed:
                results["killed"] += 1
            else:
                results["survived"] += 1
        
        results["total"] = min(len(mutants), 20)
        
        return results
    
    def _generate_python_mutants(self, code: str) -> list:
        """Generate mutants for Python code."""
        mutants = []
        
        mutations = [
            (r'==', '!='), (r'!=', '=='),
            (r'<=', '>'), (r'>=', '<'),
            (r'<(?!=)', '>='), (r'>(?!=)', '<='),
            (r'\+(?!=)', '-'), (r'-(?!=)', '+'),
            (r'\*(?!=)', '/'), (r'/(?!=)', '*'),
            (r'\band\b', 'or'), (r'\bor\b', 'and'),
            (r'\bTrue\b', 'False'), (r'\bFalse\b', 'True'),
            (r'\bnot\s+', ''),
            (r'is\s+None', 'is not None'),
            (r'is\s+not\s+None', 'is None'),
        ]
        
        for pattern, replacement in mutations:
            for match in re.finditer(pattern, code):
                mutant = code[:match.start()] + replacement + code[match.end():]
                if mutant != code:
                    mutants.append(mutant)
        
        return mutants
    
    def _test_python_mutant(
        self, 
        source_path: Path, 
        mutant_code: str, 
        original_code: str, 
        test_path: Path, 
        test_dir: Path
    ) -> bool:
        """Test if a mutant is killed. Returns True if killed."""
        # Apply mutant
        source_path.write_text(mutant_code, encoding='utf-8')
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_dir)
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path.name, "-x", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                env=env,
                timeout=30
            )
            killed = result.returncode != 0
        except:
            killed = True
        finally:
            # Restore original
            source_path.write_text(original_code, encoding='utf-8')
        
        return killed
    
    def _run_java_benchmark(
        self,
        bot: BotInterface,
        project_path: Path,
        project_name: str,
        verbose: bool,
        debug: bool
    ) -> List[FileResult]:
        """Run per-file benchmark for Java (placeholder)."""
        if verbose:
            print("    Java per-file benchmark not yet implemented")
        return []
    
    def _run_javascript_benchmark(
        self,
        bot: BotInterface,
        project_path: Path,
        project_name: str,
        verbose: bool,
        debug: bool
    ) -> List[FileResult]:
        """Run per-file benchmark for JavaScript (placeholder)."""
        if verbose:
            print("    JavaScript per-file benchmark not yet implemented")
        return []
    
    def _print_project_summary(self, results: List[FileResult]):
        """Print summary for a project."""
        if not results:
            return
        
        print(f"\n{'─'*40}")
        print("PROJECT SUMMARY:")
        
        total_coverage = sum(r.line_coverage_pct for r in results) / len(results) if results else 0
        total_mutation = sum(r.mutation_score_pct for r in results) / len(results) if results else 0
        files_with_errors = sum(1 for r in results if r.had_errors)
        files_fixed = sum(1 for r in results if r.errors_fixed)
        files_with_remaining_errors = sum(1 for r in results if r.errors_remain)
        
        print(f"  Files tested:       {len(results)}")
        print(f"  Avg Coverage:       {total_coverage:.1f}%")
        print(f"  Avg Mutation Score: {total_mutation:.1f}%")
        print(f"  Files with errors:  {files_with_errors}")
        print(f"  Files fixed:        {files_fixed}")
        print(f"  Errors remaining:   {files_with_remaining_errors}")
    
    def run_all_bots(
        self,
        project_path: str,
        language: str,
        project_name: str = None,
        verbose: bool = True,
        debug: bool = False
    ) -> List[FileResult]:
        """Run benchmark on all available bots."""
        all_results = []
        for bot_id, display_name in get_available_bots():
            try:
                bot = create_bot(bot_id, debug=debug)
                results = self.run_benchmark(
                    bot=bot,
                    project_path=project_path,
                    language=language,
                    project_name=project_name,
                    verbose=verbose,
                    debug=debug
                )
                all_results.extend(results)
            except Exception as e:
                print(f"Error with {display_name}: {e}")
        return all_results
    
    def save_results(self) -> dict:
        """Save all results to CSV files."""
        import csv
        
        files = {}
        
        # benchmark_summary.csv - one row per file
        summary_path = self.output_dir / "benchmark_summary.csv"
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "bot_name", "language", "project", "file_name", "timestamp",
                "line_coverage_pct", "tests_passed", "tests_failed",
                "mutation_score_pct", "mutants_killed", "mutants_survived",
                "errors_fixed_pct", "errors_remain"
            ])
            for r in self.results:
                errors_fixed_pct = 100.0 if r.errors_fixed else (0.0 if r.had_errors else "N/A")
                writer.writerow([
                    r.bot_name, r.language, r.project_name, r.file_name, r.timestamp,
                    round(r.line_coverage_pct, 2), r.tests_passed, r.tests_failed,
                    round(r.mutation_score_pct, 2), r.mutants_killed, r.mutants_survived,
                    errors_fixed_pct, r.errors_remain
                ])
        files["summary"] = str(summary_path)
        
        # creation_metrics.csv
        creation_path = self.output_dir / "creation_metrics.csv"
        with open(creation_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "bot_name", "language", "project", "file_name",
                "test_generated", "generation_time_sec",
                "line_coverage_pct", "tests_passed", "tests_failed"
            ])
            for r in self.results:
                writer.writerow([
                    r.bot_name, r.language, r.project_name, r.file_name,
                    r.test_generated, round(r.generation_time_seconds, 2),
                    round(r.line_coverage_pct, 2), r.tests_passed, r.tests_failed
                ])
        files["creation"] = str(creation_path)
        
        # execution_metrics.csv
        execution_path = self.output_dir / "execution_metrics.csv"
        with open(execution_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "bot_name", "language", "project", "file_name",
                "tests_passed", "tests_failed",
                "mutants_total", "mutants_killed", "mutants_survived", "mutation_score_pct"
            ])
            for r in self.results:
                writer.writerow([
                    r.bot_name, r.language, r.project_name, r.file_name,
                    r.tests_passed, r.tests_failed,
                    r.mutants_total, r.mutants_killed, r.mutants_survived,
                    round(r.mutation_score_pct, 2)
                ])
        files["execution"] = str(execution_path)
        
        # maintenance_metrics.csv
        maintenance_path = self.output_dir / "maintenance_metrics.csv"
        with open(maintenance_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "bot_name", "language", "project", "file_name",
                "had_errors", "errors_fixed", "errors_remain",
                "fix_attempts", "first_attempt_fix"
            ])
            for r in self.results:
                writer.writerow([
                    r.bot_name, r.language, r.project_name, r.file_name,
                    r.had_errors, r.errors_fixed, r.errors_remain,
                    r.fix_attempts, r.first_attempt_fix
                ])
        files["maintenance"] = str(maintenance_path)
        
        return files


def discover_projects(language: str, base_dir: Path = None) -> list:
    """Discover all projects for a given language."""
    if base_dir is None:
        base_dir = Path(".")
    
    lang_dirs = {
        "python": "python",
        "java": "java",
        "javascript": "js"
    }
    
    lang_dir = base_dir / lang_dirs.get(language, language)
    
    if not lang_dir.exists():
        return []
    
    projects = []
    
    for item in lang_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            src_dir = item / "src"
            if src_dir.exists() and src_dir.is_dir():
                projects.append(src_dir)
            elif (item / "lib").exists():
                projects.append(item / "lib")
            else:
                extensions = {
                    "python": "*.py",
                    "java": "*.java",
                    "javascript": "*.js"
                }
                ext = extensions.get(language, "*.*")
                if list(item.glob(ext)):
                    projects.append(item)
    
    return sorted(projects)


def main():
    parser = argparse.ArgumentParser(description="AI Testing Bot Benchmark")
    parser.add_argument("--bot", type=str, choices=["chatgpt", "claude", "gemini", "all"],
                        help="Bot to benchmark (or 'all' to run all bots)")
    parser.add_argument("--language", type=str, required=True,
                        choices=["python", "java", "javascript", "all"], 
                        help="Programming language")
    parser.add_argument("--project", type=str, default=None,
                        help="Path to source code")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    run_all_bots = False
    if args.bot and args.bot.lower() == "all":
        run_all_bots = True
        args.bot = None
    
    if args.debug:
        print(f"[DEBUG] Args: bot={args.bot}, run_all_bots={run_all_bots}, language={args.language}, project={args.project}")
    
    if not args.bot and not run_all_bots:
        parser.error("Specify --bot <bot_name> or --bot all")
    
    if args.language.lower() == "all":
        languages = ["python", "java", "javascript"]
    else:
        languages = [args.language]
    
    runner = BenchmarkRunner(output_dir=args.output)
    
    for language in languages:
        print(f"\n{'#'*60}")
        print(f"# LANGUAGE: {language.upper()}")
        print(f"{'#'*60}")
        
        if args.project:
            project_paths = [Path(args.project)]
            if not project_paths[0].exists():
                print(f"Project path does not exist: {args.project}")
                continue
        else:
            project_paths = discover_projects(language)
            if not project_paths:
                print(f"No projects found in ./{language}/ directory. Skipping.")
                continue
            print(f"\nDiscovered {len(project_paths)} project(s):")
            for p in project_paths:
                print(f"  - {p}")
        
        for project_path in project_paths:
            if args.debug:
                print(f"\n[DEBUG] Processing project: {project_path}")
            
            if run_all_bots:
                runner.run_all_bots(
                    project_path=str(project_path),
                    language=language,
                    verbose=not args.quiet,
                    debug=args.debug
                )
            else:
                bot = create_bot(args.bot, debug=args.debug)
                runner.run_benchmark(
                    bot=bot,
                    project_path=str(project_path),
                    language=language,
                    verbose=not args.quiet,
                    debug=args.debug
                )
    
    # Save results
    if runner.results:
        csv_files = runner.save_results()
        print(f"\n{'='*60}")
        print("CSV FILES GENERATED:")
        for name, path in csv_files.items():
            print(f"  {name}: {path}")
        print(f"{'='*60}")
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()