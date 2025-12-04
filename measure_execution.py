#!/usr/bin/env python3
"""
Test Execution Measurement Module
Measures how well AI-generated test suites execute and detect faults.

Key Metrics:
- Fault detection rate (using mutation testing or known bug injection)
- Test suite execution time and performance
- Flaky test detection
- Test isolation and independence
- False positive/negative rates
"""

import subprocess
import time
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ExecutionMeasurement:
    """Measurement results for test execution."""
    bot_name: str
    language: str
    project_name: str
    
    # Basic info
    total_test_cases: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    total_execution_time_seconds: float = 0.0
    
    # Mutation Testing Results
    total_mutants: int = 0
    mutants_killed: int = 0
    mutants_survived: int = 0  # These are false negatives
    mutation_score: float = 0.0  # killed / total (higher is better)
    
    # Mutation details
    mutation_details: list = None
    
    def __post_init__(self):
        if self.mutation_details is None:
            self.mutation_details = []
    
    def calculate_mutation_score(self):
        """Calculate mutation score (percentage of mutants killed)."""
        if self.total_mutants > 0:
            self.mutation_score = (self.mutants_killed / self.total_mutants) * 100
    
    def get_false_negatives(self) -> int:
        """Surviving mutants are false negatives - bugs the tests didn't catch."""
        return self.mutants_survived
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def get_score(self) -> float:
        """Return mutation score (higher is better - more bugs caught)."""
        return round(self.mutation_score, 2)


class TestExecutionMeasurer:
    """Measures test execution and fault detection capabilities using mutation testing."""
    
    # Mutation operators for Python
    MUTATION_OPERATORS = [
        # Arithmetic operators
        (r'(\s)\+(\s)', r'\1-\2', 'arithmetic: + to -'),
        (r'(\s)-(\s)', r'\1+\2', 'arithmetic: - to +'),
        (r'(\s)\*(\s)', r'\1/\2', 'arithmetic: * to /'),
        (r'(\s)/(\s)', r'\1*\2', 'arithmetic: / to *'),
        
        # Comparison operators
        (r'==', r'!=', 'comparison: == to !='),
        (r'!=', r'==', 'comparison: != to =='),
        (r'<=', r'>', 'comparison: <= to >'),
        (r'>=', r'<', 'comparison: >= to <'),
        (r'(?<![<>!=])>', r'<', 'comparison: > to <'),
        (r'(?<![<>!=])<', r'>', 'comparison: < to >'),
        
        # Logical operators
        (r'\band\b', r'or', 'logical: and to or'),
        (r'\bor\b', r'and', 'logical: or to and'),
        (r'\bnot\b', r'', 'logical: remove not'),
        
        # Boolean literals
        (r'\bTrue\b', r'False', 'boolean: True to False'),
        (r'\bFalse\b', r'True', 'boolean: False to True'),
        
        # Boundary mutations
        (r'\b0\b', r'1', 'boundary: 0 to 1'),
        (r'\b1\b', r'0', 'boundary: 1 to 0'),
        
        # Return value mutations
        (r'return\s+(.+)', r'return None', 'return: value to None'),
    ]
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_python(
        self,
        bot_name: str,
        project_path: str,
        test_path: str,
        project_name: str = "unnamed",
        max_mutants: int = 20,
        debug: bool = False
    ) -> ExecutionMeasurement:
        """
        Measure test execution using mutation testing.
        
        Args:
            bot_name: Name of the AI bot
            project_path: Path to source code
            test_path: Path to test suite
            project_name: Name of the project
            max_mutants: Maximum number of mutants to generate (for speed)
            debug: Enable debug output
        """
        measurement = ExecutionMeasurement(
            bot_name=bot_name,
            language="python",
            project_name=project_name
        )
        
        project_path = Path(project_path).resolve()
        test_path = Path(test_path).resolve()
        
        # First, run tests normally to get baseline
        if debug:
            print(f"    [DEBUG] Running baseline tests...")
        
        baseline = self._run_python_tests(project_path, test_path, debug)
        measurement.total_test_cases = baseline["total"]
        measurement.tests_passed = baseline["passed"]
        measurement.tests_failed = baseline["failed"]
        measurement.total_execution_time_seconds = baseline["duration"]
        
        if debug:
            print(f"    [DEBUG] Baseline: {baseline['passed']} passed, {baseline['failed']} failed")
        
        # Only run mutation testing if baseline tests pass
        if baseline["passed"] == 0:
            if debug:
                print(f"    [DEBUG] No passing tests, skipping mutation testing")
            return measurement
        
        # Run mutation testing
        if debug:
            print(f"    [DEBUG] Starting mutation testing...")
        
        mutation_results = self._run_mutation_testing(
            project_path, test_path, max_mutants, debug
        )
        
        measurement.total_mutants = mutation_results["total"]
        measurement.mutants_killed = mutation_results["killed"]
        measurement.mutants_survived = mutation_results["survived"]
        measurement.mutation_details = mutation_results["details"]
        measurement.calculate_mutation_score()
        
        if debug:
            print(f"    [DEBUG] Mutation score: {measurement.mutation_score:.1f}%")
            print(f"    [DEBUG] Killed: {measurement.mutants_killed}, Survived: {measurement.mutants_survived}")
        
        return measurement
    
    def _run_mutation_testing(
        self, 
        project_path: Path, 
        test_path: Path, 
        max_mutants: int,
        debug: bool = False
    ) -> dict:
        """Run mutation testing on the source code."""
        results = {
            "total": 0,
            "killed": 0,
            "survived": 0,
            "details": []
        }
        
        # Find all Python source files
        source_files = list(project_path.glob("*.py"))
        source_files = [f for f in source_files if not f.name.startswith("test_")]
        
        if not source_files:
            return results
        
        # Create mutation working directory
        mutation_dir = self.work_dir / "mutations"
        mutation_dir.mkdir(exist_ok=True)
        
        mutants_tested = 0
        
        for source_file in source_files:
            if mutants_tested >= max_mutants:
                break
                
            original_code = source_file.read_text(encoding='utf-8')
            
            # Generate mutants for this file
            mutants = self._generate_mutants(original_code, max_mutants - mutants_tested)
            
            for mutant_code, mutation_desc, line_num in mutants:
                if mutants_tested >= max_mutants:
                    break
                
                mutants_tested += 1
                results["total"] += 1
                
                if debug:
                    print(f"    [DEBUG] Testing mutant {mutants_tested}: {mutation_desc} at line {line_num}")
                
                # Test this mutant
                killed = self._test_mutant(
                    source_file, mutant_code, test_path, mutation_dir, debug
                )
                
                if killed:
                    results["killed"] += 1
                    status = "killed"
                else:
                    results["survived"] += 1
                    status = "survived"
                
                results["details"].append({
                    "file": source_file.name,
                    "line": line_num,
                    "mutation": mutation_desc,
                    "status": status
                })
        
        return results
    
    def _generate_mutants(self, code: str, max_count: int) -> list:
        """Generate mutants from source code."""
        mutants = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and empty lines
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Skip import statements
            if stripped.startswith('import ') or stripped.startswith('from '):
                continue
            
            # Try each mutation operator
            for pattern, replacement, desc in self.MUTATION_OPERATORS:
                if len(mutants) >= max_count:
                    return mutants
                
                if re.search(pattern, line):
                    # Create mutated line
                    mutated_line = re.sub(pattern, replacement, line, count=1)
                    
                    if mutated_line != line:
                        # Create full mutated code
                        mutated_lines = lines.copy()
                        mutated_lines[line_num - 1] = mutated_line
                        mutated_code = '\n'.join(mutated_lines)
                        
                        # Verify it's valid Python
                        try:
                            compile(mutated_code, '<string>', 'exec')
                            mutants.append((mutated_code, desc, line_num))
                        except SyntaxError:
                            pass  # Invalid mutation, skip
        
        return mutants
    
    def _test_mutant(
        self, 
        source_file: Path, 
        mutant_code: str, 
        test_path: Path,
        mutation_dir: Path,
        debug: bool = False
    ) -> bool:
        """Test a single mutant. Returns True if killed (tests fail), False if survived."""
        
        # Copy test files and source to mutation directory
        # Clear previous files
        for f in mutation_dir.glob("*.py"):
            f.unlink()
        
        # Copy test files and fix imports
        for test_file in test_path.glob("test_*.py"):
            test_content = test_file.read_text(encoding='utf-8')
            # Fix imports - replace "from src.X import" with "from X import"
            # This handles cases like "from src.slugify import ..." -> "from slugify import ..."
            test_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
            (mutation_dir / test_file.name).write_text(test_content, encoding='utf-8')
        
        # Copy other source files (non-mutated)
        for src_file in source_file.parent.glob("*.py"):
            if src_file.name != source_file.name and not src_file.name.startswith("test_"):
                shutil.copy(src_file, mutation_dir / src_file.name)
        
        # Write mutated source file
        mutant_file = mutation_dir / source_file.name
        mutant_file.write_text(mutant_code, encoding='utf-8')
        
        # Run tests
        env = os.environ.copy()
        env["PYTHONPATH"] = str(mutation_dir)
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", ".", "-x", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                cwd=str(mutation_dir),
                env=env,
                timeout=30
            )
            
            # If tests fail (return code != 0), the mutant was killed
            return result.returncode != 0
            
        except subprocess.TimeoutExpired:
            # Timeout counts as killed (mutant caused infinite loop or hang)
            return True
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Error testing mutant: {e}")
            return True  # Error counts as killed
    
    def _run_python_tests(self, project_path: Path, test_path: Path, debug: bool = False) -> dict:
        """Run pytest and return results."""
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "duration": 0.0
        }
        
        # Copy source files to test directory
        for src_file in project_path.glob("*.py"):
            if not src_file.name.startswith("test_"):
                shutil.copy(src_file, test_path / src_file.name)
        
        # Fix imports in test files - replace "from src.X import" with "from X import"
        for test_file in test_path.glob("test_*.py"):
            test_content = test_file.read_text(encoding='utf-8')
            fixed_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
            if fixed_content != test_content:
                test_file.write_text(fixed_content, encoding='utf-8')
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_path)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", ".", "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=str(test_path),
                env=env,
                timeout=120
            )
            
            results["duration"] = time.time() - start_time
            output = result.stdout + result.stderr
            
            if debug:
                print(f"    [DEBUG] pytest output:\n{output[-500:]}")
            
            # Parse results
            passed = re.search(r"(\d+) passed", output)
            failed = re.search(r"(\d+) failed", output)
            error = re.search(r"(\d+) error", output)
            
            results["passed"] = int(passed.group(1)) if passed else 0
            results["failed"] = int(failed.group(1)) if failed else 0
            results["errors"] = int(error.group(1)) if error else 0
            results["total"] = results["passed"] + results["failed"] + results["errors"]
            
        except subprocess.TimeoutExpired:
            results["duration"] = time.time() - start_time
            if debug:
                print("    [DEBUG] pytest timed out")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] pytest error: {e}")
        
        return results


if __name__ == "__main__":
    print("Test Execution Measurer")
    print("Measures test quality using mutation testing")
    print("Usage: measurer.measure_python(bot_name, project_path, test_path)")