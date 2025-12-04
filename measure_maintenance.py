#!/usr/bin/env python3
"""
Test Maintenance Measurement Module

Measures how well AI bots can fix broken tests that they generated.
Takes the test errors from the creation/execution phase and asks the AI to fix them.

Key Metric: Percentage of errors fixed
"""

import subprocess
import time
import re
import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Callable


@dataclass
class BrokenTest:
    """Represents a broken test from the generated test file."""
    file_path: str
    test_name: str
    code: str
    error_type: str
    error_message: str


@dataclass
class MaintenanceMeasurement:
    """Measurement results for test maintenance."""
    bot_name: str
    language: str
    project_name: str
    
    # Key metrics
    total_broken_tests: int = 0
    successful_repairs: int = 0
    failed_repairs: int = 0
    
    # Attempt tracking
    total_repair_attempts: int = 0  # Total attempts across all repairs
    first_attempt_fixes: int = 0    # Fixed on first try
    
    # Supporting info
    total_repair_time_seconds: float = 0.0
    repair_details: list = None
    
    def __post_init__(self):
        if self.repair_details is None:
            self.repair_details = []
    
    def get_fix_percentage(self) -> float | None:
        """Return percentage of errors fixed, or None if no errors."""
        if self.total_broken_tests == 0:
            return None  # N/A - no errors to fix
        return round((self.successful_repairs / self.total_broken_tests) * 100, 2)
    
    def get_efficiency_score(self) -> float | None:
        """
        Return efficiency score based on attempts needed.
        100% = all fixed on first attempt
        Lower = needed multiple attempts
        None = no repairs attempted
        """
        if self.successful_repairs == 0:
            return None
        # Efficiency = first_attempt_fixes / successful_repairs * 100
        return round((self.first_attempt_fixes / self.successful_repairs) * 100, 2)
    
    def get_avg_attempts_per_fix(self) -> float | None:
        """Return average attempts needed per successful fix."""
        if self.successful_repairs == 0:
            return None
        return round(self.total_repair_attempts / self.successful_repairs, 2)
    
    def get_fix_percentage_display(self) -> str:
        """Return percentage as string, or 'N/A' if no errors."""
        pct = self.get_fix_percentage()
        if pct is None:
            return "N/A"
        return f"{pct:.1f}%"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def get_score(self) -> float | None:
        """Return percentage of errors fixed, or None if no errors."""
        return self.get_fix_percentage()


class TestMaintenanceMeasurer:
    """Measures test maintenance capabilities by fixing actual test errors."""
    
    MAX_REPAIR_ATTEMPTS = 3  # Maximum attempts to fix a single test
    
    def __init__(self, work_dir: str = None):
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def measure_python(
        self,
        bot_name: str,
        repair_function: Callable[[str, str, str], str],
        test_dir: str,
        source_dir: str,
        project_name: str = "unnamed",
        debug: bool = False
    ) -> MaintenanceMeasurement:
        """
        Measure test maintenance for Python with retry logic.
        
        Args:
            bot_name: Name of the AI bot
            repair_function: Function(code, error_message, error_type) -> fixed_code
            test_dir: Path to generated tests
            source_dir: Path to source code
            project_name: Name of the project
            debug: Enable debug output
        """
        measurement = MaintenanceMeasurement(
            bot_name=bot_name,
            language="python",
            project_name=project_name
        )
        
        test_dir = Path(test_dir)
        source_dir = Path(source_dir)
        
        # First check for syntax errors in test files
        broken_tests = self._find_syntax_errors(test_dir, debug)
        
        # Then check for runtime errors by running pytest
        if not broken_tests:
            broken_tests = self._find_broken_tests_python(test_dir, source_dir, debug)
        
        measurement.total_broken_tests = len(broken_tests)
        
        if not broken_tests:
            if debug:
                print("    [DEBUG] No broken tests found")
            return measurement
        
        print(f"    Found {len(broken_tests)} broken test(s)")
        
        # Try to fix each broken test with retries
        for broken_test in broken_tests:
            current_code = broken_test.code
            current_error = broken_test.error_message
            current_error_type = broken_test.error_type
            
            fixed = False
            attempts = 0
            start_time = time.time()
            
            while not fixed and attempts < self.MAX_REPAIR_ATTEMPTS:
                attempts += 1
                measurement.total_repair_attempts += 1
                
                if debug:
                    print(f"    [DEBUG] Attempt {attempts}/{self.MAX_REPAIR_ATTEMPTS} to fix: {broken_test.test_name} ({current_error_type})")
                
                try:
                    # Call the bot's repair function
                    fixed_code = repair_function(
                        current_code,
                        current_error,
                        current_error_type
                    )
                    
                    # Validate the fix by running the test
                    is_valid, new_error, new_error_type = self._validate_and_get_error(
                        fixed_code, broken_test, source_dir, debug
                    )
                    
                    if is_valid:
                        fixed = True
                        repair_time = time.time() - start_time
                        measurement.total_repair_time_seconds += repair_time
                        measurement.successful_repairs += 1
                        
                        if attempts == 1:
                            measurement.first_attempt_fixes += 1
                        
                        measurement.repair_details.append({
                            "test": broken_test.test_name,
                            "error_type": broken_test.error_type,
                            "status": "fixed",
                            "attempts": attempts,
                            "time": repair_time
                        })
                        
                        # Save the fixed code back to the original file
                        try:
                            test_file = Path(broken_test.file_path)
                            test_file.write_text(fixed_code, encoding='utf-8')
                            if debug:
                                print(f"    [DEBUG] Saved fixed code to: {test_file}")
                        except Exception as save_error:
                            if debug:
                                print(f"    [DEBUG] Warning: Could not save fixed code: {save_error}")
                        
                        if debug:
                            print(f"    [DEBUG] Successfully fixed: {broken_test.test_name} (attempt {attempts})")
                    else:
                        # Update for next attempt
                        current_code = fixed_code
                        current_error = new_error
                        current_error_type = new_error_type
                        
                        if debug:
                            print(f"    [DEBUG] Fix attempt {attempts} failed, new error: {new_error_type}")
                            
                except Exception as e:
                    if debug:
                        print(f"    [DEBUG] Error during repair attempt {attempts}: {e}")
            
            if not fixed:
                repair_time = time.time() - start_time
                measurement.total_repair_time_seconds += repair_time
                measurement.failed_repairs += 1
                measurement.repair_details.append({
                    "test": broken_test.test_name,
                    "error_type": broken_test.error_type,
                    "status": "failed",
                    "attempts": attempts,
                    "time": repair_time
                })
                if debug:
                    print(f"    [DEBUG] Failed to fix after {attempts} attempts: {broken_test.test_name}")
        
        return measurement
    
    def _validate_and_get_error(
        self, 
        fixed_code: str, 
        broken_test: BrokenTest, 
        source_dir: Path, 
        debug: bool = False
    ) -> tuple[bool, str, str]:
        """
        Validate fix and return (is_valid, error_message, error_type).
        If valid, error_message and error_type are empty strings.
        """
        if not fixed_code or not fixed_code.strip():
            return False, "Empty code", "empty_error"
        
        # Check syntax first
        try:
            compile(fixed_code, '<string>', 'exec')
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} at line {e.lineno}", "syntax_error"
        
        # For syntax errors in the original, just checking syntax is enough
        if broken_test.error_type == "syntax_error":
            return True, "", ""
        
        # Write to temp file and run pytest
        temp_file = self.work_dir / f"test_fix_{broken_test.test_name}.py"
        temp_file.write_text(fixed_code, encoding='utf-8')
        
        # Copy source files to work dir
        for src_file in source_dir.glob("*.py"):
            if not src_file.name.startswith("test_"):
                shutil.copy(src_file, self.work_dir / src_file.name)
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.work_dir)
            
            result = subprocess.run(
                ["python", "-m", "pytest", str(temp_file), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.work_dir),
                env=env
            )
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0 or "passed" in output.lower():
                return True, "", ""
            
            # Extract error from output
            error_type, error_message = self._extract_error_info(output, broken_test.test_name)
            return False, error_message, error_type
            
        except subprocess.TimeoutExpired:
            return False, "Test timed out", "timeout_error"
        except Exception as e:
            return False, str(e), "unknown_error"
    
    def _find_syntax_errors(self, test_dir: Path, debug: bool = False) -> list[BrokenTest]:
        """Find syntax errors in test files by trying to compile them."""
        broken_tests = []
        
        test_files = list(test_dir.glob("test_*.py"))
        
        for test_file in test_files:
            try:
                code = test_file.read_text(encoding='utf-8')
                compile(code, str(test_file), 'exec')
            except SyntaxError as e:
                if debug:
                    print(f"    [DEBUG] Syntax error in {test_file.name}: {e}")
                
                broken_tests.append(BrokenTest(
                    file_path=str(test_file),
                    test_name=test_file.name,
                    code=code,
                    error_type="syntax_error",
                    error_message=f"SyntaxError: {e.msg} at line {e.lineno}"
                ))
        
        return broken_tests
    
    def _find_broken_tests_python(self, test_dir: Path, source_dir: Path, debug: bool = False) -> list[BrokenTest]:
        """Find all broken tests by running pytest and collecting errors."""
        broken_tests = []
        
        test_dir = test_dir.resolve()
        source_dir = source_dir.resolve()
        
        test_files = list(test_dir.glob("test_*.py"))
        if not test_files:
            return broken_tests
        
        # Copy source files to test directory so imports work
        for src_file in source_dir.glob("*.py"):
            if not src_file.name.startswith("test_"):
                shutil.copy(src_file, test_dir / src_file.name)
        
        # Run pytest from the test directory
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(test_dir)
            
            result = subprocess.run(
                [
                    "python", "-m", "pytest", 
                    ".", 
                    "-v", 
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(test_dir),
                env=env
            )
            
            output = result.stdout + result.stderr
            
            if debug:
                print(f"    [DEBUG] pytest output for finding broken tests:\n{output[-800:]}")
            
            # Parse the output for failures
            broken_tests = self._parse_pytest_failures(output, test_dir, debug)
            
        except subprocess.TimeoutExpired:
            if debug:
                print("    [DEBUG] pytest timed out")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Error running pytest: {e}")
        
        return broken_tests
    
    def _parse_pytest_failures(self, output: str, test_dir: Path, debug: bool = False) -> list[BrokenTest]:
        """Parse pytest output to extract failing tests that have actual errors (not assertion failures)."""
        broken_tests = []
        
        # Error types that indicate broken tests (test code bugs, not assertion failures)
        REPAIRABLE_ERRORS = [
            'syntaxerror', 'syntax_error',
            'importerror', 'import_error', 
            'modulenotfounderror',
            'nameerror', 'name_error',
            'typeerror', 'type_error',
            'attributeerror', 'attribute_error',
            'indentationerror',
            'keyerror', 'key_error',
            'indexerror', 'index_error',
        ]
        
        # Look for FAILED or ERROR lines
        # Format: FAILED test_file.py::test_name - ErrorType: message
        # Also handles: FAILED test_file.py::ClassName::test_name
        failure_pattern = r'(FAILED|ERROR)\s+([^\s:]+)::(\S+)'
        
        matches = list(re.finditer(failure_pattern, output))
        
        if debug:
            print(f"    [DEBUG] Found {len(matches)} total failures in pytest output")
        
        for match in matches:
            status = match.group(1)
            file_name = match.group(2).strip()
            test_name = match.group(3).strip()
            
            # Handle class::method format
            if '::' in test_name:
                test_name = test_name.split('::')[-1]
            
            # Extract error info from output
            error_type, error_message = self._extract_error_info(output, test_name)
            
            # Skip assertion errors - those are expected test behavior
            if 'assertion' in error_type.lower() or error_type.lower() == 'assertionerror':
                if debug:
                    print(f"    [DEBUG] Skipping {test_name} - AssertionError (test working correctly)")
                continue
            
            # Only include tests with repairable errors
            is_repairable = any(err in error_type.lower() for err in REPAIRABLE_ERRORS)
            
            if not is_repairable:
                if debug:
                    print(f"    [DEBUG] Skipping {test_name} - {error_type} (not a repairable error)")
                continue
            
            if debug:
                print(f"    [DEBUG] Found repairable {status}: {file_name}::{test_name} ({error_type})")
            
            # Find the test file
            test_file = test_dir / file_name
            if not test_file.exists():
                # Try just the filename
                test_file = test_dir / Path(file_name).name
            
            if not test_file.exists():
                if debug:
                    print(f"    [DEBUG] Test file not found: {file_name}")
                continue
            
            # Read the test file content
            try:
                full_code = test_file.read_text(encoding='utf-8')
            except Exception as e:
                if debug:
                    print(f"    [DEBUG] Could not read {file_name}: {e}")
                continue
            
            # Extract the specific test function
            test_code = self._extract_test_function(full_code, test_name)
            if not test_code:
                test_code = full_code  # Use full file if can't extract
            
            broken_tests.append(BrokenTest(
                file_path=str(test_file),
                test_name=test_name,
                code=test_code,
                error_type=error_type,
                error_message=error_message
            ))
        
        if debug:
            print(f"    [DEBUG] Repairable broken tests: {len(broken_tests)}")
        
        return broken_tests
    
    def _extract_test_function(self, code: str, test_name: str) -> str:
        """Extract a specific test function from the code."""
        lines = code.split('\n')
        result = []
        in_function = False
        indent_level = 0
        
        # Also collect imports at the top
        imports = []
        for line in lines:
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
            elif line.strip() and not line.startswith('#'):
                break
        
        for i, line in enumerate(lines):
            if f'def {test_name}' in line:
                in_function = True
                indent_level = len(line) - len(line.lstrip())
                result.append(line)
            elif in_function:
                if line.strip() == '':
                    result.append(line)
                elif line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                    result.append(line)
                elif line.strip().startswith('def ') or line.strip().startswith('class '):
                    break
                elif not line.strip():
                    result.append(line)
                else:
                    break
        
        if result:
            return '\n'.join(imports + [''] + result)
        return None
    
    def _extract_error_info(self, output: str, test_name: str) -> tuple[str, str]:
        """Extract error type and message for a specific test."""
        error_type = "unknown_error"
        error_message = "Test failed"
        
        # Common error patterns
        patterns = [
            (r'(\w+Error):\s*(.+)', 'error'),
            (r'(\w+Exception):\s*(.+)', 'exception'),
            (r'(AssertionError):\s*(.+)', 'assertion'),
            (r'assert\s+(.+)', 'assertion'),
        ]
        
        # Find error near the test name in output
        test_section = output
        if test_name in output:
            # Get the section around this test
            idx = output.find(test_name)
            test_section = output[idx:idx+1000]
        
        for pattern, _ in patterns:
            match = re.search(pattern, test_section)
            if match:
                error_type = match.group(1).lower()
                if len(match.groups()) > 1:
                    error_message = match.group(2)[:200]  # Limit message length
                break
        
        return error_type, error_message


if __name__ == "__main__":
    print("Test Maintenance Measurer")
    print("Measures how well AI bots fix their own broken tests")