#!/usr/bin/env python3
"""
Test Creation Measurement Module
Measures how well AI bots generate tests for given source code.

Metrics collected:
- Code coverage (line, branch, statement, function)
- Compilation success rate
- Test pass rate
- Assertion quality and density
- Generation time and efficiency
"""

import subprocess
import time
import json
import os
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Callable
import shutil


@dataclass
class CreationMeasurement:
    """Measurement results for test creation."""
    bot_name: str
    language: str
    project_name: str
    
    # Source info
    source_files_count: int = 0
    
    # Generation info
    tests_generated: int = 0
    test_files_created: int = 0
    generation_time_seconds: float = 0.0
    
    # THE KEY METRIC: Coverage
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    
    # Supporting info
    tests_compilable: int = 0
    tests_passing: int = 0
    compilation_errors: list = None
    
    def __post_init__(self):
        if self.compilation_errors is None:
            self.compilation_errors = []
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def get_score(self) -> float:
        """Return line coverage as the score."""
        return round(self.line_coverage, 2)


class TestCreationMeasurer:
    """Measures test creation capabilities of AI bots."""
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_python(
        self,
        bot_name: str,
        project_path: str,
        generate_tests: Callable[[str, str], str],
        project_name: str = "unnamed",
        debug: bool = False
    ) -> CreationMeasurement:
        """
        Measure test creation for a Python project.
        
        Args:
            bot_name: Name of the AI bot
            project_path: Path to source code directory
            generate_tests: Function that takes (source_code, module_name) and returns generated tests
            project_name: Name of the project being tested
            debug: Enable debug output
        """
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="python",
            project_name=project_name
        )
        
        project_path = Path(project_path)
        
        # Count source files
        source_files = list(project_path.rglob("*.py"))
        source_files = [f for f in source_files if not f.name.startswith("test_")]
        measurement.source_files_count = len(source_files)
        
        if debug:
            print(f"    [DEBUG] Found {len(source_files)} source files")
        
        # Create test output directory
        test_output_dir = self.work_dir / "generated_tests"
        test_output_dir.mkdir(exist_ok=True)
        
        # Generate tests for each source file
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text(encoding='utf-8')
            module_name = source_file.stem  # e.g., "slugify" from "slugify.py"
            
            try:
                generated_test = generate_tests(source_code, module_name)
                
                if generated_test:
                    # Save generated test
                    test_filename = f"test_{source_file.stem}.py"
                    test_path = test_output_dir / test_filename
                    test_path.write_text(generated_test, encoding='utf-8')
                    measurement.test_files_created += 1
                    print(f"    Generated: {test_filename}")
                    
            except Exception as e:
                print(f"Error generating test for {source_file}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Analyze generated tests
        self._analyze_python_tests(measurement, project_path, test_output_dir, debug)
        
        return measurement
    
    def _analyze_python_tests(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Analyze generated Python tests."""
        import ast
        
        test_files = list(test_dir.glob("test_*.py"))
        measurement.tests_generated = len(test_files)
        
        if debug:
            print(f"    [DEBUG] Found {len(test_files)} test files")
        
        # Check compilation and count tests
        for test_file in test_files:
            content = test_file.read_text(encoding='utf-8')
            
            try:
                tree = ast.parse(content)
                measurement.tests_compilable += 1
                    
            except SyntaxError as e:
                measurement.compilation_errors.append({
                    "file": test_file.name,
                    "error": str(e)
                })
                if debug:
                    print(f"    [DEBUG] Syntax error in {test_file.name}: {e}")
        
        # Run tests and collect coverage
        self._run_python_coverage(measurement, project_path, test_dir, debug)
    
    def _count_python_assertions(self, code: str) -> int:
        """Count assertions in Python code."""
        patterns = [
            r'\bassert\s+',
            r'self\.assert\w+\(',
            r'pytest\.raises\(',
            r'\.assert_called',
            r'assertEqual\(',
            r'assertTrue\(',
            r'assertFalse\(',
            r'assertIs\(',
            r'assertIn\(',
        ]
        return sum(len(re.findall(p, code)) for p in patterns)
    
    def _run_python_coverage(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Run pytest with coverage and update measurement."""
        import json
        
        # Make sure paths are absolute
        test_dir = test_dir.resolve()
        project_path = project_path.resolve()
        
        # Get list of source files
        source_files = [f for f in project_path.glob("*.py") if not f.name.startswith("test_")]
        
        if not source_files:
            print("    Warning: No source files found for coverage")
            return
        
        # Copy source files to test directory so imports work
        # Also fix imports in test files
        for src_file in source_files:
            shutil.copy(src_file, test_dir / src_file.name)
            if debug:
                print(f"    [DEBUG] Copied {src_file.name} to test dir")
        
        # Fix imports in test files - replace "from src.X import" with "from X import"
        for test_file in test_dir.glob("test_*.py"):
            test_content = test_file.read_text(encoding='utf-8')
            fixed_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
            if fixed_content != test_content:
                test_file.write_text(fixed_content, encoding='utf-8')
                if debug:
                    print(f"    [DEBUG] Fixed imports in {test_file.name}")
        
        # Remove any existing pytest.ini that might interfere
        pytest_ini = test_dir / "pytest.ini"
        if pytest_ini.exists():
            pytest_ini.unlink()
        
        # Run pytest with coverage on specific source files only (not test files)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_dir)
        
        # Build coverage source argument - only measure the source files we copied
        source_file_names = [f.stem for f in source_files]  # e.g., ["slugify"]
        cov_sources = ",".join(source_file_names)
        
        if debug:
            print(f"    [DEBUG] Source files: {[f.name for f in source_files]}")
            print(f"    [DEBUG] Coverage sources: {cov_sources}")
            print(f"    [DEBUG] Test dir (absolute): {test_dir}")
            test_files = list(test_dir.glob('test_*.py'))
            print(f"    [DEBUG] Test files: {test_files}")
            
            # Show first few lines of test file to verify it has test functions
            if test_files:
                content = test_files[0].read_text(encoding='utf-8')
                lines = content.split('\n')[:30]
                print(f"    [DEBUG] First 30 lines of {test_files[0].name}:")
                for i, line in enumerate(lines):
                    print(f"    [DEBUG]   {i+1}: {line}")
        
        try:
            # Run pytest from the test directory, measuring coverage only on source files
            cmd = [
                "python", "-m", "pytest",
                ".",
                f"--cov={cov_sources}",
                "--cov-report=json",
                "--cov-report=term",
                "-v", "--tb=short"
            ]
            
            if debug:
                print(f"    [DEBUG] Running from {test_dir}: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                env=env,
                timeout=300
            )
            
            # Parse test results
            output = result.stdout + result.stderr
            
            if debug:
                print(f"    [DEBUG] pytest return code: {result.returncode}")
                print(f"    [DEBUG] pytest output (last 1500 chars):")
                print(output[-1500:] if len(output) > 1500 else output)
            
            passed = re.search(r"(\d+) passed", output)
            failed = re.search(r"(\d+) failed", output)
            error = re.search(r"(\d+) error", output)
            
            measurement.tests_passing = int(passed.group(1)) if passed else 0
            
            # Parse coverage from terminal output as backup
            # Look for pattern like "TOTAL    100     20    80%"
            cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            if cov_match:
                measurement.line_coverage = float(cov_match.group(1))
                if debug:
                    print(f"    [DEBUG] Coverage from terminal: {measurement.line_coverage}%")
            
            # Try to parse coverage.json
            coverage_file = test_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, encoding='utf-8') as f:
                    cov_data = json.load(f)
                    totals = cov_data.get("totals", {})
                    if debug:
                        print(f"    [DEBUG] coverage.json totals: {totals}")
                    if totals.get("percent_covered"):
                        measurement.line_coverage = totals.get("percent_covered", 0.0)
                    if totals.get("percent_covered_branches"):
                        measurement.branch_coverage = totals.get("percent_covered_branches", 0.0)
            else:
                if debug:
                    print(f"    [DEBUG] coverage.json not found at {coverage_file}")
                        
        except subprocess.TimeoutExpired:
            print("    Warning: Coverage timed out")
        except Exception as e:
            print(f"    Warning: Coverage error: {e}")
            if debug:
                import traceback
                traceback.print_exc()
    
    def _analyze_test_naming(self, test_files: list[Path]) -> float:
        """
        Analyze quality of test function names.
        Good names: test_function_does_something_when_condition
        Bad names: test1, testA, test_it
        """
        good_patterns = 0
        total_patterns = 0
        
        for tf in test_files:
            content = tf.read_text()
            test_funcs = re.findall(r'def (test_\w+)\(', content)
            
            for func_name in test_funcs:
                total_patterns += 1
                # Check for descriptive naming
                words = func_name.split('_')
                if len(words) >= 3:  # test_something_action
                    good_patterns += 1
        
        return good_patterns / max(total_patterns, 1)
    
    def measure_java(
        self,
        bot_name: str,
        project_path: str,
        generate_tests: Callable[[str], str],
        project_name: str = "unnamed"
    ) -> CreationMeasurement:
        """Measure test creation for a Java project."""
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="java",
            project_name=project_name
        )
        
        project_path = Path(project_path)
        
        # Find Java source files
        source_files = list(project_path.rglob("*.java"))
        source_files = [f for f in source_files if "Test" not in f.name]
        measurement.source_files_count = len(source_files)
        measurement.source_loc = sum(
            len(f.read_text().splitlines()) for f in source_files
        )
        
        # Create test output directory
        test_output_dir = self.work_dir / "generated_tests" / "java"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text()
            
            try:
                generated_test = generate_tests(source_code)
                
                if generated_test:
                    test_filename = f"{source_file.stem}Test.java"
                    test_path = test_output_dir / test_filename
                    test_path.write_text(generated_test)
                    measurement.test_files_created += 1
                    
            except Exception as e:
                print(f"Error generating test for {source_file}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Analyze Java tests
        self._analyze_java_tests(measurement, project_path, test_output_dir)
        
        measurement.calculate_rates()
        return measurement
    
    def _analyze_java_tests(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path
    ):
        """Analyze generated Java tests."""
        test_files = list(test_dir.glob("*Test.java"))
        measurement.tests_generated = len(test_files)
        
        for test_file in test_files:
            content = test_file.read_text()
            measurement.test_loc += len(content.splitlines())
            
            # Check for basic syntax (simplified)
            if "class" in content and "@Test" in content:
                measurement.tests_compilable += 1
            
            # Count assertions
            assertion_patterns = [
                r'assert\w+\(',
                r'Assertions\.\w+\(',
                r'assertThat\(',
                r'verify\(',
            ]
            assertion_count = sum(len(re.findall(p, content)) for p in assertion_patterns)
            measurement.total_assertions += assertion_count
            if assertion_count > 0:
                measurement.tests_with_assertions += 1
            
            # Count test methods
            test_methods = len(re.findall(r'@Test\s+.*?void\s+\w+', content, re.DOTALL))
            
            # Quality indicators
            if "@ParameterizedTest" in content:
                measurement.has_edge_cases = True
            if "assertThrows" in content or "expectedException" in content.lower():
                measurement.has_error_handling_tests = True
    
    def measure_javascript(
        self,
        bot_name: str,
        project_path: str,
        generate_tests: Callable[[str], str],
        project_name: str = "unnamed"
    ) -> CreationMeasurement:
        """Measure test creation for a JavaScript project."""
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="javascript",
            project_name=project_name
        )
        
        project_path = Path(project_path)
        
        # Find JS source files
        source_files = list(project_path.rglob("*.js"))
        source_files.extend(project_path.rglob("*.ts"))
        source_files = [f for f in source_files if ".test." not in f.name and ".spec." not in f.name]
        measurement.source_files_count = len(source_files)
        measurement.source_loc = sum(
            len(f.read_text().splitlines()) for f in source_files
        )
        
        # Create test output directory
        test_output_dir = self.work_dir / "generated_tests" / "js"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text()
            
            try:
                generated_test = generate_tests(source_code)
                
                if generated_test:
                    test_filename = f"{source_file.stem}.test.js"
                    test_path = test_output_dir / test_filename
                    test_path.write_text(generated_test)
                    measurement.test_files_created += 1
                    
            except Exception as e:
                print(f"Error generating test for {source_file}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Analyze JS tests
        self._analyze_js_tests(measurement, test_output_dir)
        
        measurement.calculate_rates()
        return measurement
    
    def _analyze_js_tests(self, measurement: CreationMeasurement, test_dir: Path):
        """Analyze generated JavaScript tests."""
        test_files = list(test_dir.glob("*.test.js"))
        test_files.extend(test_dir.glob("*.spec.js"))
        measurement.tests_generated = len(test_files)
        
        for test_file in test_files:
            content = test_file.read_text()
            measurement.test_loc += len(content.splitlines())
            
            # Check syntax with Node
            result = subprocess.run(
                ["node", "--check", str(test_file)],
                capture_output=True
            )
            if result.returncode == 0:
                measurement.tests_compilable += 1
            else:
                measurement.compilation_errors.append({
                    "file": test_file.name,
                    "error": result.stderr.decode()
                })
            
            # Count assertions
            assertion_patterns = [
                r'expect\(',
                r'\.toBe\(',
                r'\.toEqual\(',
                r'\.toThrow\(',
                r'\.toHaveBeenCalled',
                r'assert\.',
            ]
            assertion_count = sum(len(re.findall(p, content)) for p in assertion_patterns)
            measurement.total_assertions += assertion_count
            if assertion_count > 0:
                measurement.tests_with_assertions += 1
            
            # Quality indicators
            if "describe.each" in content or "it.each" in content:
                measurement.has_edge_cases = True
            if ".toThrow" in content or ".rejects" in content:
                measurement.has_error_handling_tests = True


def print_measurement_report(measurement: CreationMeasurement):
    """Print a formatted measurement report."""
    print("\n" + "=" * 60)
    print(f"TEST CREATION MEASUREMENT REPORT")
    print(f"Bot: {measurement.bot_name}")
    print(f"Language: {measurement.language}")
    print(f"Project: {measurement.project_name}")
    print("=" * 60)
    
    print(f"\n📁 SOURCE CODE:")
    print(f"   Files: {measurement.source_files_count}")
    print(f"   Lines of Code: {measurement.source_loc:,}")
    
    print(f"\n🧪 GENERATED TESTS:")
    print(f"   Test Files Created: {measurement.test_files_created}")
    print(f"   Test Lines of Code: {measurement.test_loc:,}")
    print(f"   Generation Time: {measurement.generation_time_seconds:.2f}s")
    
    print(f"\n✅ COMPILATION:")
    print(f"   Compilable: {measurement.tests_compilable}/{measurement.tests_generated}")
    print(f"   Compilation Rate: {measurement.compilation_rate:.1%}")
    if measurement.compilation_errors:
        print(f"   Errors: {len(measurement.compilation_errors)}")
    
    print(f"\n🏃 EXECUTION:")
    print(f"   Passing: {measurement.tests_passing}")
    print(f"   Failing: {measurement.tests_failing}")
    print(f"   Erroring: {measurement.tests_erroring}")
    print(f"   Pass Rate: {measurement.pass_rate:.1%}")
    
    print(f"\n📊 COVERAGE:")
    print(f"   Line Coverage: {measurement.line_coverage:.1f}%")
    print(f"   Branch Coverage: {measurement.branch_coverage:.1f}%")
    
    print(f"\n🎯 ASSERTION QUALITY:")
    print(f"   Total Assertions: {measurement.total_assertions}")
    print(f"   Tests with Assertions: {measurement.tests_with_assertions}")
    print(f"   Avg Assertions/Test: {measurement.avg_assertions_per_test:.1f}")
    
    print(f"\n🔍 QUALITY INDICATORS:")
    print(f"   Edge Case Tests: {'✓' if measurement.has_edge_cases else '✗'}")
    print(f"   Error Handling Tests: {'✓' if measurement.has_error_handling_tests else '✗'}")
    print(f"   Boundary Tests: {'✓' if measurement.has_boundary_tests else '✗'}")
    print(f"   Test Naming Quality: {measurement.test_naming_quality:.1%}")
    
    print(f"\n⭐ OVERALL SCORE: {measurement.get_score()}/100")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage with a mock bot
    def mock_test_generator(source_code: str) -> str:
        """Mock test generator for demonstration."""
        return """
import pytest

def test_example():
    assert True

def test_another_example():
    result = 1 + 1
    assert result == 2
"""
    
    measurer = TestCreationMeasurer()
    
    print("Test Creation Measurer initialized.")
    print("Use measurer.measure_python(), measure_java(), or measure_javascript()")
    print("to measure test generation capabilities.")