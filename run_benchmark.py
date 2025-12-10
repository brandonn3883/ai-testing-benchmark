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
                timeout=120,
                encoding='utf-8',
                errors='replace'
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
        mutants_to_test = mutants[:20]
        for i, mutant_code in enumerate(mutants_to_test):
            if debug:
                print(f"        [DEBUG] Testing mutant {i+1}/{len(mutants_to_test)}...", end=" ", flush=True)
            
            killed = self._test_python_mutant(
                source_in_test_dir, mutant_code, original_code,
                test_path, test_dir
            )
            if killed:
                results["killed"] += 1
                if debug:
                    print("killed")
            else:
                results["survived"] += 1
                if debug:
                    print("survived")
        
        results["total"] = len(mutants_to_test)
        
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
        
        killed = True  # Default to killed (safer assumption on timeout/error)
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path.name, "-x", "--tb=no", "-q", "--timeout=10"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                env=env,
                timeout=15,  # Shorter timeout
                encoding='utf-8',
                errors='replace'
            )
            killed = result.returncode != 0
        except subprocess.TimeoutExpired:
            # Timeout means mutant likely caused infinite loop - count as killed
            killed = True
        except Exception:
            # Any other error - count as killed
            killed = True
        finally:
            # Always restore original
            try:
                source_path.write_text(original_code, encoding='utf-8')
            except:
                pass
        
        return killed
    
    def _run_java_benchmark(
        self,
        bot: BotInterface,
        project_path: Path,
        project_name: str,
        verbose: bool,
        debug: bool
    ) -> List[FileResult]:
        """Run per-file benchmark for Java."""
        
        # Check if Java/javac is available
        javac_cmd = "javac"
        java_cmd = "java"
        
        try:
            subprocess.run([javac_cmd, "-version"], capture_output=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            if verbose:
                print("    ✗ Java compiler (javac) not found. Please install JDK.")
                print("    Skipping Java benchmark.")
            return []
        
        # Find all source files
        all_source_files = list(project_path.rglob("*.java"))
        # Exclude test files
        all_source_files = [f for f in all_source_files 
                           if not f.name.endswith('Test.java') 
                           and not f.name.endswith('Tests.java')
                           and not f.name.startswith('Test')
                           and 'test' not in str(f).lower().split(os.sep)]
        
        # Files to test
        testable_files = [f for f in all_source_files 
                         if not f.name.startswith('.')]
        
        if verbose:
            print(f"\n    Found {len(testable_files)} testable file(s)")
        
        if not testable_files:
            if verbose:
                print("    No Java files found to test")
            return []
        
        # Create work directory
        test_dir = self.work_dir / "generated_tests"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ALL source files
        if debug:
            print(f"    [DEBUG] Copying source files to {test_dir}")
        
        # Extract package info and copy files
        package_info = {}  # filename -> package name
        for src_file in all_source_files:
            content = src_file.read_text(encoding='utf-8', errors='replace')
            # Extract package declaration
            package_match = re.search(r'package\s+([\w.]+)\s*;', content)
            package_name = package_match.group(1) if package_match else ""
            package_info[src_file.name] = package_name
            
            # Copy file (remove package declaration for flat structure)
            content_no_package = re.sub(r'package\s+[\w.]+\s*;', '', content)
            dest_file = test_dir / src_file.name
            dest_file.write_text(content_no_package, encoding='utf-8')
            if debug:
                print(f"    [DEBUG]   Copied: {src_file.name} (package: {package_name})")
        
        if debug:
            print(f"    [DEBUG] Files in test_dir: {[f.name for f in test_dir.glob('*.java')]}")
        
        # Download JUnit if not present
        junit_jar = test_dir / "junit-platform-console-standalone.jar"
        jacoco_agent = test_dir / "jacocoagent.jar"
        jacoco_cli = test_dir / "jacococli.jar"
        
        if not junit_jar.exists():
            if verbose:
                print("    Downloading JUnit...")
            try:
                # Download JUnit standalone
                junit_url = "https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar"
                self._download_file(junit_url, junit_jar)
                if verbose:
                    print("    JUnit downloaded successfully")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not download JUnit: {e}")
                    print("    Skipping Java benchmark.")
                return []
        
        if not jacoco_agent.exists():
            if verbose:
                print("    Downloading JaCoCo...")
            try:
                # Download JaCoCo agent
                jacoco_agent_url = "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.11/org.jacoco.agent-0.8.11-runtime.jar"
                self._download_file(jacoco_agent_url, jacoco_agent)
                # Download JaCoCo CLI
                jacoco_cli_url = "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.11/org.jacoco.cli-0.8.11-nodeps.jar"
                self._download_file(jacoco_cli_url, jacoco_cli)
                if verbose:
                    print("    JaCoCo downloaded successfully")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not download JaCoCo: {e}")
        
        # Scan source files for external dependencies and download them
        external_jars = self._download_java_dependencies(test_dir, all_source_files, verbose, debug)
        
        # Build class contents map for context
        class_contents = {}
        for sf in all_source_files:
            class_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        # Process each file
        results = []
        for idx, source_file in enumerate(testable_files):
            if verbose:
                print(f"\n    [{idx + 1}/{len(testable_files)}] {source_file.name}")
            
            result = self._process_java_file(
                bot=bot,
                source_file=source_file,
                test_dir=test_dir,
                project_name=project_name,
                class_contents=class_contents,
                all_source_files=all_source_files,
                junit_jar=junit_jar,
                jacoco_agent=jacoco_agent,
                jacoco_cli=jacoco_cli,
                external_jars=external_jars,
                verbose=verbose,
                debug=debug
            )
            results.append(result)
            self.results.append(result)
        
        # Print summary
        if verbose:
            self._print_project_summary(results)
        
        return results
    
    def _download_file(self, url: str, dest: Path):
        """Download a file from URL."""
        import urllib.request
        urllib.request.urlretrieve(url, str(dest))
    
    def _download_java_dependencies(
        self, 
        test_dir: Path, 
        source_files: list, 
        verbose: bool, 
        debug: bool
    ) -> List[Path]:
        """Scan source files for external imports and download required JARs."""
        
        # Map of known imports to Maven coordinates
        dependency_map = {
            'com.ibm.icu': {
                'name': 'ICU4J',
                'jar': 'icu4j.jar',
                'url': 'https://repo1.maven.org/maven2/com/ibm/icu/icu4j/74.2/icu4j-74.2.jar'
            },
            'lombok': {
                'name': 'Lombok',
                'jar': 'lombok.jar',
                'url': 'https://repo1.maven.org/maven2/org/projectlombok/lombok/1.18.30/lombok-1.18.30.jar'
            },
            'org.apache.commons.lang3': {
                'name': 'Apache Commons Lang',
                'jar': 'commons-lang3.jar',
                'url': 'https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.14.0/commons-lang3-3.14.0.jar'
            },
            'org.apache.commons.io': {
                'name': 'Apache Commons IO',
                'jar': 'commons-io.jar',
                'url': 'https://repo1.maven.org/maven2/commons-io/commons-io/2.15.1/commons-io-2.15.1.jar'
            },
            'com.google.gson': {
                'name': 'Gson',
                'jar': 'gson.jar',
                'url': 'https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar'
            },
            'com.fasterxml.jackson': {
                'name': 'Jackson',
                'jar': 'jackson-databind.jar',
                'url': 'https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.16.1/jackson-databind-2.16.1.jar'
            },
            'org.slf4j': {
                'name': 'SLF4J',
                'jar': 'slf4j-api.jar',
                'url': 'https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.11/slf4j-api-2.0.11.jar'
            },
            'com.google.guava': {
                'name': 'Guava',
                'jar': 'guava.jar',
                'url': 'https://repo1.maven.org/maven2/com/google/guava/guava/33.0.0-jre/guava-33.0.0-jre.jar'
            },
        }
        
        # Scan all source files for imports
        needed_deps = set()
        for src_file in source_files:
            try:
                content = src_file.read_text(encoding='utf-8', errors='replace')
                for import_match in re.finditer(r'import\s+([\w.]+)', content):
                    import_path = import_match.group(1)
                    # Check if this import matches any known dependency
                    for dep_prefix, dep_info in dependency_map.items():
                        if import_path.startswith(dep_prefix):
                            needed_deps.add(dep_prefix)
                            break
            except:
                pass
        
        # Download needed dependencies
        downloaded_jars = []
        for dep_prefix in needed_deps:
            dep_info = dependency_map[dep_prefix]
            jar_path = test_dir / dep_info['jar']
            
            if not jar_path.exists():
                if verbose:
                    print(f"    Downloading {dep_info['name']}...")
                try:
                    self._download_file(dep_info['url'], jar_path)
                    # Verify download succeeded
                    if jar_path.exists() and jar_path.stat().st_size > 0:
                        downloaded_jars.append(jar_path)
                        if verbose:
                            print(f"    {dep_info['name']} downloaded successfully ({jar_path.stat().st_size} bytes)")
                    else:
                        if verbose:
                            print(f"    Warning: {dep_info['name']} download failed - file empty or missing")
                except Exception as e:
                    if verbose:
                        print(f"    Warning: Could not download {dep_info['name']}: {e}")
            else:
                downloaded_jars.append(jar_path)
                if debug:
                    print(f"    [DEBUG] {dep_info['name']} already exists ({jar_path.stat().st_size} bytes)")
        
        if debug and downloaded_jars:
            print(f"    [DEBUG] External JARs to use: {[str(j.absolute()) for j in downloaded_jars]}")
        
        return downloaded_jars
    
    def _process_java_file(
        self,
        bot: BotInterface,
        source_file: Path,
        test_dir: Path,
        project_name: str,
        class_contents: dict,
        all_source_files: list,
        junit_jar: Path,
        jacoco_agent: Path,
        jacoco_cli: Path,
        external_jars: List[Path],
        verbose: bool,
        debug: bool
    ) -> FileResult:
        """Process a single Java file through the complete pipeline."""
        
        result = FileResult(
            bot_name=bot.name,
            language="java",
            project_name=project_name,
            file_name=source_file.name
        )
        
        source_code = source_file.read_text(encoding='utf-8', errors='replace')
        class_name = source_file.stem
        test_filename = f"{class_name}Test.java"
        test_path = test_dir / test_filename
        
        # ========== STEP 1: Generate Test ==========
        if verbose:
            print(f"        [1/3] Test Creation...")
        start_time = time.time()
        
        # Build context with dependencies
        code_with_context = self._build_java_context(
            source_code, class_name, class_contents, all_source_files, source_file
        )
        
        try:
            generated_test = bot.generate_tests(
                code_with_context, 
                module_name=class_name, 
                language="java"
            )
            if generated_test:
                # Remove package declaration if present
                generated_test = re.sub(r'package\s+[\w.]+\s*;', '', generated_test)
                # Ensure JUnit imports
                if 'import org.junit' not in generated_test and 'import static org.junit' not in generated_test:
                    imports = "import org.junit.jupiter.api.*;\nimport static org.junit.jupiter.api.Assertions.*;\n\n"
                    generated_test = imports + generated_test
                
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
        
        # ========== STEP 2: Compile and Run Tests ==========
        coverage, passed, failed, error_msg = self._run_java_coverage(
            test_path, class_name, test_dir, junit_jar, jacoco_agent, jacoco_cli, external_jars, debug
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
                        # Remove package declaration if present
                        fixed_test = re.sub(r'package\s+[\w.]+\s*;', '', fixed_test)
                        test_path.write_text(fixed_test, encoding='utf-8')
                        
                        # Re-run tests
                        coverage, passed, failed, error_msg = self._run_java_coverage(
                            test_path, class_name, test_dir, junit_jar, jacoco_agent, jacoco_cli, external_jars, debug
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
        
        if debug:
            print(f"              [DEBUG] Java mutation check: errors_remain={result.errors_remain}, tests_passed={result.tests_passed}")
        
        if not result.errors_remain and result.tests_passed > 0:
            mutation_results = self._run_java_mutation(
                source_file, test_path, test_dir, class_name, junit_jar, external_jars, debug
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
                    print(f"              Skipped (no passing tests: {result.tests_passed})")
        
        return result
    
    def _build_java_context(
        self, 
        source_code: str, 
        class_name: str, 
        class_contents: dict, 
        all_source_files: list,
        source_file: Path
    ) -> str:
        """Build source code with dependency context for Java."""
        # Find import statements for local classes
        imports = []
        for match in re.finditer(r'import\s+([\w.]+)\s*;', source_code):
            import_path = match.group(1)
            # Get the class name (last part)
            imported_class = import_path.split('.')[-1]
            if imported_class != '*':
                imports.append(imported_class)
        
        # Also look for class references in code
        for other_class in class_contents.keys():
            if other_class != class_name and other_class in source_code:
                if other_class not in imports:
                    imports.append(other_class)
        
        dependency_code = ""
        for imp in imports:
            if imp in class_contents and imp != class_name:
                dep_content = class_contents[imp]
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n// ... (truncated)"
                dependency_code += f"\n\n// === DEPENDENCY: {imp}.java ===\n{dep_content}"
        
        if dependency_code:
            return source_code + "\n\n// === PROJECT DEPENDENCIES ===" + dependency_code
        elif len(all_source_files) > 1:
            other_files = [sf.name for sf in all_source_files if sf != source_file]
            if other_files:
                return source_code + f"\n\n// NOTE: Project also contains: {', '.join(other_files)}"
        
        return source_code
    
    def _run_java_coverage(
        self, 
        test_path: Path, 
        class_name: str, 
        test_dir: Path,
        junit_jar: Path,
        jacoco_agent: Path,
        jacoco_cli: Path,
        external_jars: List[Path],
        debug: bool
    ) -> tuple:
        """
        Compile and run Java tests with coverage.
        Returns: (coverage_pct, tests_passed, tests_failed, error_message)
        """
        # Classpath separator
        cp_sep = ";" if os.name == 'nt' else ":"
        
        # Build classpath with external JARs (use absolute paths)
        jar_paths = [str(test_dir.absolute())]
        for j in external_jars:
            jar_paths.append(str(j.absolute()))
        base_classpath = cp_sep.join(jar_paths)
        
        if debug:
            print(f"              [DEBUG] Classpath: {base_classpath[:200]}...")
            print(f"              [DEBUG] External JARs: {[j.name for j in external_jars]}")
            for j in external_jars:
                print(f"              [DEBUG]   {j.name} exists: {j.exists()}")
        
        # Step 1: Compile source files
        source_files = list(test_dir.glob("*.java"))
        source_files = [f for f in source_files if not f.name.endswith("Test.java")]
        
        if source_files:
            # Use just filenames since we're running from test_dir
            source_filenames = [f.name for f in source_files]
            # Add -encoding UTF-8 to handle unicode characters
            compile_cmd = ["javac", "-encoding", "UTF-8", "-cp", base_classpath] + source_filenames
            
            if debug:
                print(f"              [DEBUG] Compile command: javac -encoding UTF-8 -cp <classpath> {' '.join(source_filenames)}")
            
            try:
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=60,
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode != 0:
                    if debug:
                        print(f"              [DEBUG] Source compilation failed: {result.stderr[:300]}")
                    return None, 0, 0, f"Compilation error: {result.stderr[:200]}"
            except Exception as e:
                return None, 0, 0, f"Compilation error: {e}"
        
        # Step 2: Compile test file
        classpath = f"{base_classpath}{cp_sep}{str(junit_jar.absolute())}"
        # Use just filename since we're running from test_dir
        # Add -encoding UTF-8 to handle unicode characters in test file
        compile_test_cmd = ["javac", "-encoding", "UTF-8", "-cp", classpath, test_path.name]
        
        if debug:
            print(f"              [DEBUG] Test compile classpath includes JUnit: {junit_jar.absolute()}")
            print(f"              [DEBUG] JUnit JAR exists: {junit_jar.exists()}")
        
        try:
            result = subprocess.run(
                compile_test_cmd,
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=60,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode != 0:
                error_msg = result.stderr[:300] if result.stderr else "Unknown compilation error"
                if debug:
                    print(f"              [DEBUG] Test compilation failed: {error_msg}")
                return None, 0, 0, error_msg
        except Exception as e:
            return None, 0, 0, f"Test compilation error: {e}"
        
        # Step 3: Run tests with JaCoCo coverage
        jacoco_exec = test_dir / "jacoco.exec"
        if jacoco_exec.exists():
            jacoco_exec.unlink()
        
        test_class_name = test_path.stem  # e.g., "CalculatorTest"
        
        # Use just "jacoco.exec" filename since cwd is test_dir
        run_cmd = [
            "java",
            f"-javaagent:{jacoco_agent.absolute()}=destfile=jacoco.exec",
            "-jar", str(junit_jar.absolute()),
            "--class-path", base_classpath,
            "--select-class", test_class_name
        ]
        
        try:
            result = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=120,
                encoding='utf-8',
                errors='replace'
            )
            
            output = result.stdout + result.stderr
            
            if debug:
                print(f"              [DEBUG] JUnit return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"              [DEBUG] Output (last 500): {output[-500:]}")
            
            # Parse test counts from JUnit output
            passed = 0
            failed = 0
            
            # JUnit output format: "[        58 tests successful      ]"
            # Need to handle variable whitespace
            passed_match = re.search(r'\[\s*(\d+)\s+tests?\s+successful\s*\]', output)
            failed_match = re.search(r'\[\s*(\d+)\s+tests?\s+failed\s*\]', output)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            
            if debug:
                print(f"              [DEBUG] Parsed tests: {passed} passed, {failed} failed")
            
            # Check for ACTUAL errors (not assertion failures)
            # Assertion failures (expected vs actual) are expected behavior - tests ran successfully
            # We only want to fix: compilation errors, NoClassDefFoundError, ClassNotFoundException, etc.
            error_msg = None
            if result.returncode != 0:
                # If tests actually ran (passed > 0 or failed > 0), it's not a code error
                tests_actually_ran = passed > 0 or failed > 0
                
                if not tests_actually_ran:
                    # No tests ran - likely a compilation or setup error
                    error_patterns = [
                        r'(error:\s*cannot find symbol[^\n]+)',
                        r'(error:\s*package [^\n]+ does not exist[^\n]*)',
                        r'(error:\s*class [^\n]+ not found[^\n]*)',
                        r'(Exception in thread[^\n]+)',
                        r'(NoClassDefFoundError[^\n]+)',
                        r'(ClassNotFoundException[^\n]+)',
                        r'(NoSuchMethodError[^\n]+)',
                        r'(IllegalAccessError[^\n]+)',
                    ]
                    for pattern in error_patterns:
                        error_match = re.search(pattern, output, re.IGNORECASE)
                        if error_match:
                            error_msg = error_match.group(1)
                            break
                    
                    # If no specific error found but no tests ran at all, there's likely a setup error
                    if not error_msg:
                        # Check it's not just an assertion failure message
                        if "expected:" not in output.lower() or "but was:" not in output.lower():
                            error_msg = output[-500:]
                
                if debug and error_msg:
                    print(f"              [DEBUG] Detected fixable error: {error_msg[:100]}")
            
            # Step 4: Generate coverage report
            coverage_pct = None
            if jacoco_exec.exists() and jacoco_cli.exists():
                csv_report = test_dir / "coverage.csv"
                
                # Remove old report if exists
                if csv_report.exists():
                    csv_report.unlink()
                
                # Find only the source class files (not JARs, not test classes)
                class_files = [f.name for f in test_dir.glob("*.class") 
                              if not f.name.endswith("Test.class") and not f.name.startswith("Test")]
                
                if debug:
                    print(f"              [DEBUG] JaCoCo exec exists: {jacoco_exec.exists()}, size: {jacoco_exec.stat().st_size if jacoco_exec.exists() else 0}")
                    print(f"              [DEBUG] Class files found: {class_files}")
                
                if class_files:
                    # Build command with specific class files
                    report_cmd = [
                        "java", "-jar", str(jacoco_cli.absolute()),
                        "report", "jacoco.exec",
                        "--csv", "coverage.csv"
                    ]
                    # Add each class file
                    for cf in class_files:
                        report_cmd.extend(["--classfiles", cf])
                    
                    if debug:
                        print(f"              [DEBUG] Running coverage report command...")
                    
                    try:
                        cov_result = subprocess.run(
                            report_cmd,
                            capture_output=True,
                            text=True,
                            cwd=str(test_dir),
                            timeout=30,
                            encoding='utf-8',
                            errors='replace'
                        )
                        
                        if debug:
                            print(f"              [DEBUG] Coverage report return code: {cov_result.returncode}")
                            if cov_result.stderr:
                                print(f"              [DEBUG] Coverage report stderr: {cov_result.stderr[:200]}")
                        
                        # Parse CSV for coverage
                        if csv_report.exists():
                            csv_content = csv_report.read_text(encoding='utf-8', errors='replace')
                            if debug:
                                print(f"              [DEBUG] Coverage CSV content: {csv_content[:300]}")
                            
                            lines = csv_content.strip().split('\n')
                            if len(lines) > 1:
                                # Parse header to find column indices
                                header = lines[0].split(',')
                                try:
                                    class_idx = header.index('CLASS')
                                    line_missed_idx = header.index('LINE_MISSED')
                                    line_covered_idx = header.index('LINE_COVERED')
                                    
                                    for line in lines[1:]:  # Skip header
                                        parts = line.split(',')
                                        if len(parts) > max(class_idx, line_missed_idx, line_covered_idx):
                                            csv_class = parts[class_idx]
                                            if csv_class == class_name or class_name in csv_class:
                                                missed = int(parts[line_missed_idx]) if parts[line_missed_idx].isdigit() else 0
                                                covered = int(parts[line_covered_idx]) if parts[line_covered_idx].isdigit() else 0
                                                total = missed + covered
                                                if total > 0:
                                                    coverage_pct = (covered / total) * 100
                                                    if debug:
                                                        print(f"              [DEBUG] Coverage for {csv_class}: {covered}/{total} = {coverage_pct:.1f}%")
                                                break
                                except ValueError as e:
                                    if debug:
                                        print(f"              [DEBUG] Could not find column in header: {e}")
                        else:
                            if debug:
                                print(f"              [DEBUG] Coverage CSV not created")
                    except Exception as e:
                        if debug:
                            print(f"              [DEBUG] Coverage report error: {e}")
                else:
                    if debug:
                        print(f"              [DEBUG] No class files found for coverage")
            
            return coverage_pct or 0.0, passed, failed, error_msg
            
        except subprocess.TimeoutExpired:
            return 0.0, 0, 0, "Timeout"
        except Exception as e:
            return 0.0, 0, 0, str(e)
    
    def _run_java_mutation(
        self, 
        source_file: Path, 
        test_path: Path,
        test_dir: Path, 
        class_name: str,
        junit_jar: Path,
        external_jars: List[Path],
        debug: bool
    ) -> dict:
        """Run mutation testing on a single Java file."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        source_in_test_dir = test_dir / source_file.name
        if not source_in_test_dir.exists():
            return results
        
        original_code = source_in_test_dir.read_text(encoding='utf-8', errors='replace')
        
        # Generate mutants
        mutants = self._generate_java_mutants(original_code)
        
        if debug:
            print(f"              [DEBUG] Generated {len(mutants)} mutants")
        
        # Classpath separator and build classpath
        cp_sep = ";" if os.name == 'nt' else ":"
        jar_paths = [str(test_dir)] + [str(j) for j in external_jars]
        base_classpath = cp_sep.join(jar_paths)
        
        # Test each mutant (limit to 20)
        mutants_to_test = mutants[:20]
        for i, mutant_code in enumerate(mutants_to_test):
            if debug:
                print(f"              [DEBUG] Testing mutant {i+1}/{len(mutants_to_test)}...", end=" ", flush=True)
            
            killed = self._test_java_mutant(
                source_in_test_dir, mutant_code, original_code,
                test_path, test_dir, junit_jar, base_classpath, cp_sep
            )
            if killed:
                results["killed"] += 1
                if debug:
                    print("killed")
            else:
                results["survived"] += 1
                if debug:
                    print("survived")
        
        results["total"] = len(mutants_to_test)
        
        return results
    
    def _generate_java_mutants(self, code: str) -> list:
        """Generate mutants for Java code."""
        mutants = []
        
        mutations = [
            # Comparison operators
            (r'==', '!='), (r'!=', '=='),
            (r'<=', '>'), (r'>=', '<'),
            (r'<(?!=)', '>='), (r'>(?!=)', '<='),
            # Arithmetic operators
            (r'\+(?!=)', '-'), (r'-(?!=)', '+'),
            (r'\*(?!=)', '/'), (r'/(?!=)', '*'),
            # Logical operators
            (r'&&', '||'), (r'\|\|', '&&'),
            # Boolean literals
            (r'\btrue\b', 'false'), (r'\bfalse\b', 'true'),
            # Null checks
            (r'== null', '!= null'), (r'!= null', '== null'),
            # Negation
            (r'!(?!=)', ''),
            # Increment/decrement
            (r'\+\+', '--'), (r'--', '++'),
        ]
        
        for pattern, replacement in mutations:
            for match in re.finditer(pattern, code):
                mutant = code[:match.start()] + replacement + code[match.end():]
                if mutant != code:
                    mutants.append(mutant)
        
        return mutants
    
    def _test_java_mutant(
        self, 
        source_path: Path, 
        mutant_code: str, 
        original_code: str, 
        test_path: Path, 
        test_dir: Path,
        junit_jar: Path,
        base_classpath: str,
        cp_sep: str
    ) -> bool:
        """Test if a Java mutant is killed. Returns True if killed."""
        # Apply mutant
        source_path.write_text(mutant_code, encoding='utf-8')
        
        killed = True  # Default to killed on any error
        
        try:
            # Recompile source (use filename since cwd is test_dir)
            compile_result = subprocess.run(
                ["javac", "-encoding", "UTF-8", "-cp", base_classpath, source_path.name],
                capture_output=True,
                cwd=str(test_dir),
                timeout=15,
                encoding='utf-8',
                errors='replace'
            )
            
            if compile_result.returncode != 0:
                # Mutant doesn't compile - counts as killed
                return True
            
            # Run tests
            test_class_name = test_path.stem
            run_result = subprocess.run(
                [
                    "java", "-jar", str(junit_jar.absolute()),
                    "--class-path", base_classpath,
                    "--select-class", test_class_name
                ],
                capture_output=True,
                cwd=str(test_dir),
                timeout=15,
                encoding='utf-8',
                errors='replace'
            )
            
            killed = run_result.returncode != 0
        except subprocess.TimeoutExpired:
            killed = True
        except Exception:
            killed = True
        finally:
            # Restore original and recompile
            try:
                source_path.write_text(original_code, encoding='utf-8')
                subprocess.run(
                    ["javac", "-encoding", "UTF-8", "-cp", base_classpath, source_path.name],
                    capture_output=True,
                    cwd=str(test_dir),
                    timeout=15,
                    encoding='utf-8',
                    errors='replace'
                )
            except:
                pass
        
        return killed
    
    def _run_javascript_benchmark(
        self,
        bot: BotInterface,
        project_path: Path,
        project_name: str,
        verbose: bool,
        debug: bool
    ) -> List[FileResult]:
        """Run per-file benchmark for JavaScript."""
        
        # Find all source files
        all_source_files = list(project_path.rglob("*.js"))
        # Exclude test files and node_modules
        all_source_files = [f for f in all_source_files 
                           if not f.name.endswith('.test.js') 
                           and not f.name.endswith('.spec.js')
                           and not f.name.startswith('test')
                           and 'node_modules' not in str(f)
                           and '__tests__' not in str(f)]
        
        # Files to test (exclude config files and index.js if it's just exports)
        testable_files = [f for f in all_source_files 
                         if not f.name.startswith('.')
                         and f.name not in ('jest.config.js', 'babel.config.js', 'webpack.config.js')]
        
        if verbose:
            print(f"\n    Found {len(testable_files)} testable file(s)")
        
        if not testable_files:
            if verbose:
                print("    No JavaScript files found to test")
            return []
        
        # Create work directory
        test_dir = self.work_dir / "generated_tests"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ALL source files
        if debug:
            print(f"    [DEBUG] Copying source files to {test_dir}")
        for src_file in all_source_files:
            content = src_file.read_text(encoding='utf-8', errors='replace')
            # Fix relative imports to work in flat directory
            # require('./module') -> require('./module')  (keep as-is, they'll be in same dir)
            # require('../module') -> require('./module')
            content = re.sub(r"require\s*\(\s*['\"]\.\.\/(\w+)['\"]\s*\)", r"require('./\1')", content)
            content = re.sub(r"from\s+['\"]\.\.\/(\w+)['\"]", r"from './\1'", content)
            dest_file = test_dir / src_file.name
            dest_file.write_text(content, encoding='utf-8')
            if debug:
                print(f"    [DEBUG]   Copied: {src_file.name}")
        
        if debug:
            print(f"    [DEBUG] Files in test_dir: {[f.name for f in test_dir.glob('*.js')]}")
        
        # Create package.json for Jest with ES Module support
        package_json = {
            "name": "benchmark-tests",
            "version": "1.0.0",
            "scripts": {
                "test": "jest --coverage --coverageReporters=json-summary --coverageReporters=text"
            },
            "devDependencies": {
                "jest": "^29.0.0",
                "@babel/core": "^7.0.0",
                "@babel/preset-env": "^7.0.0",
                "babel-jest": "^29.0.0"
            }
        }
        (test_dir / "package.json").write_text(json.dumps(package_json, indent=2), encoding='utf-8')
        
        # Create jest.config.js with babel transform
        jest_config = """module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: ['*.js', '!*.test.js', '!*.spec.js', '!jest.config.js', '!babel.config.js'],
  testMatch: ['**/*.test.js', '**/*.spec.js'],
  verbose: true,
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  transformIgnorePatterns: []
};
"""
        (test_dir / "jest.config.js").write_text(jest_config, encoding='utf-8')
        
        # Create babel.config.js for ES Module support
        babel_config = """module.exports = {
  presets: [
    ['@babel/preset-env', { targets: { node: 'current' } }]
  ]
};
"""
        (test_dir / "babel.config.js").write_text(babel_config, encoding='utf-8')
        
        # Check if Node.js/npm is available
        npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
        npx_cmd = "npx.cmd" if os.name == 'nt' else "npx"
        
        try:
            subprocess.run([npm_cmd, "--version"], capture_output=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            if verbose:
                print("    ✗ Node.js/npm not found. Please install Node.js from https://nodejs.org/")
                print("    Skipping JavaScript benchmark.")
            return []
        
        # Install Jest and Babel if needed (check if node_modules exists)
        node_modules = test_dir / "node_modules"
        if not node_modules.exists():
            if verbose:
                print("    Installing Jest and Babel...")
            try:
                result = subprocess.run(
                    [npm_cmd, "install", "--save-dev", "jest", "@babel/core", "@babel/preset-env", "babel-jest"],
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=180,  # Longer timeout for more packages
                    shell=(os.name == 'nt'),  # Use shell on Windows
                    encoding='utf-8',
                    errors='replace'
                )
                if result.returncode != 0:
                    if verbose:
                        print(f"    Warning: npm install failed: {result.stderr[:200] if result.stderr else 'Unknown error'}")
                elif verbose:
                    print("    Jest and Babel installed successfully")
            except subprocess.TimeoutExpired:
                if verbose:
                    print("    Warning: npm install timed out")
            except Exception as e:
                if verbose:
                    print(f"    Warning: Could not install dependencies: {e}")
        
        # Build module contents map for context
        module_contents = {}
        for sf in all_source_files:
            module_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        # Process each file
        results = []
        for idx, source_file in enumerate(testable_files):
            if verbose:
                print(f"\n    [{idx + 1}/{len(testable_files)}] {source_file.name}")
            
            result = self._process_javascript_file(
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
    
    def _process_javascript_file(
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
        """Process a single JavaScript file through the complete pipeline."""
        
        result = FileResult(
            bot_name=bot.name,
            language="javascript",
            project_name=project_name,
            file_name=source_file.name
        )
        
        source_code = source_file.read_text(encoding='utf-8', errors='replace')
        module_name = source_file.stem
        test_filename = f"{module_name}.test.js"
        test_path = test_dir / test_filename
        
        # ========== STEP 1: Generate Test ==========
        if verbose:
            print(f"        [1/3] Test Creation...")
        start_time = time.time()
        
        # Build context with dependencies
        code_with_context = self._build_javascript_context(
            source_code, module_name, module_contents, all_source_files, source_file
        )
        
        try:
            generated_test = bot.generate_tests(
                code_with_context, 
                module_name=module_name, 
                language="javascript"
            )
            if generated_test:
                # Check if source uses ES Modules
                source_in_dir = test_dir / source_file.name
                source_content = source_in_dir.read_text(encoding='utf-8', errors='replace') if source_in_dir.exists() else source_code
                uses_es_modules = 'export ' in source_content or 'export default' in source_content
                
                # Fix imports in generated test
                if uses_es_modules:
                    # Convert require to import if source uses ES modules
                    generated_test = re.sub(
                        rf"const\s+(\w+)\s*=\s*require\s*\(\s*['\"]\.?\/?{module_name}['\"]\s*\)",
                        f"import {module_name} from './{module_name}'",
                        generated_test
                    )
                    # Add import if not present
                    if f"from './{module_name}'" not in generated_test and f'from "./{module_name}"' not in generated_test:
                        if "import " not in generated_test or module_name not in generated_test:
                            generated_test = f"import {module_name} from './{module_name}';\n\n" + generated_test
                else:
                    # Ensure require statement for CommonJS
                    if f"require('./{module_name}')" not in generated_test and f"from './{module_name}'" not in generated_test:
                        if "require(" not in generated_test and "import " not in generated_test:
                            generated_test = f"const {module_name} = require('./{module_name}');\n\n" + generated_test
                
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
        coverage, passed, failed, error_msg = self._run_javascript_coverage(
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
                        coverage, passed, failed, error_msg = self._run_javascript_coverage(
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
        
        if debug:
            print(f"              [DEBUG] JS mutation check: errors_remain={result.errors_remain}, tests_passed={result.tests_passed}")
        
        if not result.errors_remain and result.tests_passed > 0:
            mutation_results = self._run_javascript_mutation(
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
                    print(f"              Skipped (no passing tests: {result.tests_passed})")
        
        return result
    
    def _build_javascript_context(
        self, 
        source_code: str, 
        module_name: str, 
        module_contents: dict, 
        all_source_files: list,
        source_file: Path
    ) -> str:
        """Build source code with dependency context for JavaScript."""
        # Find require/import statements
        imports = []
        # require('./module') or require('../module')
        for match in re.finditer(r"require\s*\(\s*['\"]\.\.?\/(\w+)['\"]", source_code):
            imports.append(match.group(1))
        # import x from './module'
        for match in re.finditer(r"from\s+['\"]\.\.?\/(\w+)['\"]", source_code):
            imports.append(match.group(1))
        
        dependency_code = ""
        for imp in imports:
            if imp in module_contents and imp != module_name:
                dep_content = module_contents[imp]
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n// ... (truncated)"
                dependency_code += f"\n\n// === DEPENDENCY: {imp}.js ===\n{dep_content}"
        
        if dependency_code:
            return source_code + "\n\n// === PROJECT DEPENDENCIES ===" + dependency_code
        elif len(all_source_files) > 1:
            other_files = [sf.name for sf in all_source_files if sf != source_file]
            if other_files:
                return source_code + f"\n\n// NOTE: Project also contains: {', '.join(other_files)}"
        
        return source_code
    
    def _run_javascript_coverage(
        self, 
        test_path: Path, 
        module_name: str, 
        test_dir: Path, 
        debug: bool
    ) -> tuple:
        """
        Run Jest with coverage.
        Returns: (coverage_pct, tests_passed, tests_failed, error_message)
        """
        # Clean old coverage
        coverage_dir = test_dir / "coverage"
        if coverage_dir.exists():
            shutil.rmtree(coverage_dir)
        
        # Use correct command for Windows vs Unix
        npx_cmd = "npx.cmd" if os.name == 'nt' else "npx"
        
        # Run Jest on just this test file
        cmd = [npx_cmd, "jest", test_path.name, "--coverage", "--coverageReporters=json-summary", "--coverageReporters=text", "--no-cache"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=120,
                shell=(os.name == 'nt'),  # Use shell on Windows
                encoding='utf-8',
                errors='replace'
            )
            
            output = result.stdout + result.stderr
            
            if debug:
                print(f"              [DEBUG] Jest return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"              [DEBUG] Output (last 500): {output[-500:]}")
            
            # Parse test counts from Jest output
            passed = 0
            failed = 0
            
            # Jest output format:
            # "Test Suites: 1 failed, 1 total"
            # "Tests:       22 failed, 35 passed, 57 total"
            # We want the "Tests:" line, not "Test Suites:"
            
            # Find the Tests: line specifically
            tests_line_match = re.search(r'Tests:\s+(.+)', output)
            if tests_line_match:
                tests_line = tests_line_match.group(1)
                # Parse passed from this line
                passed_match = re.search(r'(\d+)\s+passed', tests_line)
                if passed_match:
                    passed = int(passed_match.group(1))
                # Parse failed from this line
                failed_match = re.search(r'(\d+)\s+failed', tests_line)
                if failed_match:
                    failed = int(failed_match.group(1))
            
            if debug:
                print(f"              [DEBUG] Parsed tests: {passed} passed, {failed} failed")
            
            # Check for ACTUAL errors (not assertion failures)
            # Assertion failures (expect(...).toBe(...) failing) are expected behavior
            # We only want to fix: SyntaxError, ReferenceError, TypeError, import errors, etc.
            error_msg = None
            if result.returncode != 0:
                # Only trigger maintenance for real errors, not test assertion failures
                # If tests ran (passed > 0 or "Tests:" in output with failures), it's not a code error
                tests_actually_ran = passed > 0 or (failed > 0 and "expect(" in output.lower())
                
                if not tests_actually_ran:
                    # Look for actual code/import errors (not assertion failures)
                    error_patterns = [
                        r'(SyntaxError[^\n]+)',
                        r'(ReferenceError[^\n]+)',
                        r'(TypeError:\s*\w+\s+is not a function[^\n]*)',  # Only "is not a function" type errors
                        r'(TypeError:\s*Cannot read propert[^\n]*)',  # Property access errors
                        r'(TypeError:\s*\w+\s+is not defined[^\n]*)',
                        r'(Cannot find module[^\n]+)',
                        r'(Module not found[^\n]+)',
                        r'(Error:\s*Cannot find module[^\n]+)',
                    ]
                    
                    for pattern in error_patterns:
                        error_match = re.search(pattern, output, re.IGNORECASE)
                        if error_match:
                            error_msg = error_match.group(1)
                            break
                    
                    # If no specific error found but no tests ran at all, there's likely a setup error
                    if not error_msg and passed == 0 and failed == 0:
                        # Check if it's a compilation/parse error
                        if "SyntaxError" in output or "Cannot find module" in output or "is not defined" in output:
                            error_msg = output[-500:]
                
                if debug and error_msg:
                    print(f"              [DEBUG] Detected fixable error: {error_msg[:100]}")
            
            # Parse coverage from JSON summary
            coverage_pct = None
            coverage_summary = test_dir / "coverage" / "coverage-summary.json"
            
            if coverage_summary.exists():
                try:
                    with open(coverage_summary, encoding='utf-8') as f:
                        cov_data = json.load(f)
                        
                        if debug:
                            print(f"              [DEBUG] Coverage keys: {list(cov_data.keys())}")
                        
                        # Get coverage for the specific file
                        for filepath, file_data in cov_data.items():
                            if filepath == "total":
                                continue
                            # Extract just the filename from the full path for matching
                            # Handle both forward and back slashes
                            filename = filepath.replace('\\', '/').split('/')[-1]
                            filename_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
                            
                            if debug:
                                print(f"              [DEBUG] Checking {filename_without_ext} against {module_name}")
                            
                            if filename_without_ext == module_name or module_name in filepath:
                                coverage_pct = file_data.get("lines", {}).get("pct", 0.0)
                                if debug:
                                    print(f"              [DEBUG] Found coverage: {coverage_pct}%")
                                break
                        
                        # Fallback to total
                        if coverage_pct is None and "total" in cov_data:
                            coverage_pct = cov_data["total"].get("lines", {}).get("pct", 0.0)
                            if debug:
                                print(f"              [DEBUG] Using total coverage: {coverage_pct}%")
                                
                except Exception as e:
                    if debug:
                        print(f"              [DEBUG] Error reading coverage: {e}")
            
            # Fallback: parse from terminal output
            if coverage_pct is None:
                cov_match = re.search(r'All files\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)', output)
                if cov_match:
                    coverage_pct = float(cov_match.group(1))
            
            return coverage_pct or 0.0, passed, failed, error_msg
            
        except subprocess.TimeoutExpired:
            return 0.0, 0, 0, "Timeout"
        except Exception as e:
            return 0.0, 0, 0, str(e)
    
    def _run_javascript_mutation(
        self, 
        source_file: Path, 
        test_path: Path,
        test_dir: Path, 
        module_name: str, 
        debug: bool
    ) -> dict:
        """Run mutation testing on a single JavaScript file."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        source_in_test_dir = test_dir / source_file.name
        if not source_in_test_dir.exists():
            return results
        
        original_code = source_in_test_dir.read_text(encoding='utf-8', errors='replace')
        
        # Generate mutants
        mutants = self._generate_javascript_mutants(original_code)
        
        if debug:
            print(f"              [DEBUG] Generated {len(mutants)} mutants")
        
        # Test each mutant (limit to 20)
        mutants_to_test = mutants[:20]
        for i, mutant_code in enumerate(mutants_to_test):
            if debug:
                print(f"              [DEBUG] Testing mutant {i+1}/{len(mutants_to_test)}...", end=" ", flush=True)
            
            killed = self._test_javascript_mutant(
                source_in_test_dir, mutant_code, original_code,
                test_path, test_dir
            )
            if killed:
                results["killed"] += 1
                if debug:
                    print("killed")
            else:
                results["survived"] += 1
                if debug:
                    print("survived")
        
        results["total"] = len(mutants_to_test)
        
        return results
    
    def _generate_javascript_mutants(self, code: str) -> list:
        """Generate mutants for JavaScript code."""
        mutants = []
        
        mutations = [
            # Comparison operators
            (r'===', '!=='), (r'!==', '==='),
            (r'==', '!='), (r'!=', '=='),
            (r'<=', '>'), (r'>=', '<'),
            (r'<(?!=)', '>='), (r'>(?!=)', '<='),
            # Arithmetic operators
            (r'\+(?!=)', '-'), (r'-(?!=)', '+'),
            (r'\*(?!=)', '/'), (r'/(?!=)', '*'),
            # Logical operators
            (r'&&', '||'), (r'\|\|', '&&'),
            # Boolean literals
            (r'\btrue\b', 'false'), (r'\bfalse\b', 'true'),
            # Null/undefined checks
            (r'=== null', '!== null'), (r'!== null', '=== null'),
            (r'=== undefined', '!== undefined'), (r'!== undefined', '=== undefined'),
            # Negation
            (r'!(?!=)', ''),
        ]
        
        for pattern, replacement in mutations:
            for match in re.finditer(pattern, code):
                mutant = code[:match.start()] + replacement + code[match.end():]
                if mutant != code:
                    mutants.append(mutant)
        
        return mutants
    
    def _test_javascript_mutant(
        self, 
        source_path: Path, 
        mutant_code: str, 
        original_code: str, 
        test_path: Path, 
        test_dir: Path
    ) -> bool:
        """Test if a JavaScript mutant is killed. Returns True if killed."""
        # Apply mutant
        source_path.write_text(mutant_code, encoding='utf-8')
        
        # Use correct command for Windows vs Unix
        npx_cmd = "npx.cmd" if os.name == 'nt' else "npx"
        
        killed = True  # Default to killed (safer assumption on timeout/error)
        
        try:
            # Use shorter timeout (15s) and --testTimeout to limit Jest internally
            result = subprocess.run(
                [npx_cmd, "jest", test_path.name, "--no-cache", "--silent", "--testTimeout=10000", "--forceExit"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=15,  # Shorter timeout
                shell=(os.name == 'nt'),
                encoding='utf-8',
                errors='replace'
            )
            killed = result.returncode != 0
        except subprocess.TimeoutExpired:
            # Timeout means mutant likely caused infinite loop - count as killed
            killed = True
        except Exception:
            # Any other error - count as killed
            killed = True
        finally:
            # Always restore original
            try:
                source_path.write_text(original_code, encoding='utf-8')
            except:
                pass
        
        return killed
    
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