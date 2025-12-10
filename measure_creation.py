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
    
    # Per-file coverage: dict mapping filename to coverage percentage
    per_file_coverage: dict = None
    
    # Supporting info
    tests_compilable: int = 0
    tests_passing: int = 0
    tests_failed: int = 0  # Track failing tests
    compilation_errors: list = None
    
    def __post_init__(self):
        if self.compilation_errors is None:
            self.compilation_errors = []
        if self.per_file_coverage is None:
            self.per_file_coverage = {}
    
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
    
    def _extract_code_from_response(self, response: str, language: str = None) -> str:
        """Extract code from response, removing any markdown formatting."""
        if not response:
            return ""
        
        # Try to extract code from markdown code blocks
        patterns = [
            r'```(?:python|java|javascript|js|typescript|ts)?\n?(.*?)```',
            r'```\n?(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                # Return the longest match (most likely the full code)
                return max(matches, key=len).strip()
        
        # No code blocks found, return as-is (might already be clean code)
        return response.strip()
    
    def measure_python(
        self,
        bot_name: str,
        project_path: str,
        generate_tests: Callable[[str, str], str],
        project_name: str = "unnamed",
        debug: bool = False,
        fix_tests: Callable[[str, str, str], str] = None  # Optional function to fix broken tests
    ) -> CreationMeasurement:
        """
        Measure test creation for a Python project.
        
        Flow for each source file:
        1. Generate test
        2. Run coverage
        3. If error, fix the test (up to 3 attempts)
        4. Record per-file coverage
        5. Move to next file
        
        Args:
            bot_name: Name of the AI bot
            project_path: Path to source code directory
            generate_tests: Function that takes (source_code, module_name) and returns generated tests
            project_name: Name of the project being tested
            debug: Enable debug output
            fix_tests: Optional function that takes (test_code, error_message, source_code) and returns fixed test code
        """
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="python",
            project_name=project_name
        )
        
        project_path = Path(project_path)
        
        # Find all source files (including private ones for copying)
        all_source_files = list(project_path.rglob("*.py"))
        all_source_files = [f for f in all_source_files if not f.name.startswith("test_")]
        
        # Files to generate tests for (exclude __init__.py and _private.py files)
        testable_files = [f for f in all_source_files 
                         if f.name != "__init__.py" and not f.name.startswith("_")]
        
        measurement.source_files_count = len(testable_files)
        
        if debug:
            print(f"    [DEBUG] Found {len(all_source_files)} source files, {len(testable_files)} testable")
            excluded = [f.name for f in all_source_files if f not in testable_files]
            if excluded:
                print(f"    [DEBUG] Excluded from testing: {excluded}")
        
        # Create test output directory - clean it first
        test_output_dir = self.work_dir / "generated_tests"
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir(exist_ok=True)
        
        # Copy ALL source files to test directory (with relative imports fixed)
        for src_file in all_source_files:
            content = src_file.read_text(encoding='utf-8', errors='replace')
            content = re.sub(r'from\s+\.(\w+)\s+import', r'from \1 import', content)
            content = re.sub(r'from\s+\.(\w+\.\w+)\s+import', r'from \1 import', content)
            dest_file = test_output_dir / src_file.name
            dest_file.write_text(content, encoding='utf-8')
        
        # Build dependency context map
        module_contents = {}
        for sf in all_source_files:
            module_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        start_time = time.time()
        
        # Process each source file individually
        for source_file in testable_files:
            source_code = source_file.read_text(encoding='utf-8', errors='replace')
            module_name = source_file.stem
            test_filename = f"test_{module_name}.py"
            test_path = test_output_dir / test_filename
            
            print(f"    Processing: {source_file.name}")
            
            # Build context with dependencies
            code_with_context = self._build_python_context(
                source_code, module_name, module_contents, all_source_files, source_file
            )
            
            # Step 1: Generate test
            try:
                generated_test = generate_tests(code_with_context, module_name)
                if not generated_test:
                    print(f"      No test generated for {source_file.name}")
                    continue
                    
                test_path.write_text(generated_test, encoding='utf-8')
                measurement.test_files_created += 1
                
                if debug:
                    print(f"      [DEBUG] Generated {test_filename}")
                    
            except Exception as e:
                print(f"      Error generating test: {e}")
                continue
            
            # Step 2: Run coverage for this single file
            file_coverage, error_msg = self._run_single_file_coverage(
                test_path, source_file.name, test_output_dir, debug
            )
            
            # Step 3: If error and we have a fix function, try to fix
            if error_msg and fix_tests:
                if debug:
                    print(f"      [DEBUG] Test has errors, attempting fix...")
                
                for attempt in range(3):  # Up to 3 fix attempts
                    try:
                        fixed_test = fix_tests(
                            test_path.read_text(encoding='utf-8', errors='replace'),
                            error_msg,
                            source_code
                        )
                        if fixed_test:
                            test_path.write_text(fixed_test, encoding='utf-8')
                            
                            # Re-run coverage
                            file_coverage, error_msg = self._run_single_file_coverage(
                                test_path, source_file.name, test_output_dir, debug
                            )
                            
                            if not error_msg:
                                if debug:
                                    print(f"      [DEBUG] Fixed on attempt {attempt + 1}")
                                break
                    except Exception as e:
                        if debug:
                            print(f"      [DEBUG] Fix attempt {attempt + 1} failed: {e}")
            
            # Step 4: Record coverage for this file
            if file_coverage is not None:
                measurement.per_file_coverage[source_file.name] = round(file_coverage, 2)
                print(f"      Coverage: {file_coverage:.1f}%")
                measurement.tests_passing += 1
            else:
                measurement.per_file_coverage[source_file.name] = 0.0
                measurement.tests_failed += 1
                if error_msg:
                    print(f"      Error: {error_msg[:100]}...")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Calculate overall coverage (average of per-file)
        if measurement.per_file_coverage:
            measurement.line_coverage = sum(measurement.per_file_coverage.values()) / len(measurement.per_file_coverage)
        
        measurement.tests_compilable = measurement.test_files_created
        
        return measurement
    
    def _build_python_context(
        self, 
        source_code: str, 
        module_name: str, 
        module_contents: dict, 
        all_source_files: list,
        source_file: Path
    ) -> str:
        """Build source code with dependency context for AI."""
        # Find local imports
        imports = []
        for match in re.finditer(r'(?:from\s+(\w+)\s+import|^import\s+(\w+))', source_code, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                imports.append(module)
        
        # Add dependency code
        dependency_code = ""
        for imp in imports:
            if imp in module_contents and imp != module_name:
                dep_content = module_contents[imp]
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n# ... (truncated)"
                dependency_code += f"\n\n# === DEPENDENCY: {imp}.py ===\n{dep_content}"
        
        if dependency_code:
            return source_code + "\n\n# === PROJECT DEPENDENCIES (for reference) ===" + dependency_code
        elif len(all_source_files) > 1:
            other_files = [sf.name for sf in all_source_files if sf != source_file]
            if other_files:
                return source_code + f"\n\n# NOTE: This project also contains: {', '.join(other_files)}"
        
        return source_code
    
    def _run_single_file_coverage(
        self,
        test_path: Path,
        source_filename: str,
        test_dir: Path,
        debug: bool = False
    ) -> tuple:
        """
        Run coverage for a single test file against its source file.
        
        Returns:
            (coverage_percentage, error_message) - error_message is None if successful
        """
        import json
        
        source_name = Path(source_filename).stem  # e.g., "errors" from "errors.py"
        
        # Fix imports in test file
        test_content = test_path.read_text(encoding='utf-8', errors='replace')
        original = test_content
        test_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
        test_content = re.sub(r'from\s+(\w+)\.errors\s+import', r'from errors import', test_content)
        test_content = re.sub(r'from\s+(\w+)\._regex\s+import', r'from _regex import', test_content)
        if test_content != original:
            test_path.write_text(test_content, encoding='utf-8')
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(test_dir)
        
        # Run pytest with coverage on just this source file
        coverage_json = test_dir / "coverage.json"
        if coverage_json.exists():
            coverage_json.unlink()
        
        # Use just the filename since we run from test_dir
        test_filename = test_path.name
        
        # Verify test file exists
        if not test_path.exists():
            return None, f"Test file not found: {test_path}"
        
        cmd = [
            "python", "-m", "pytest",
            test_filename,
            f"--cov={source_name}",
            "--cov-report=json:coverage.json",
            "--cov-report=term",
            "-v", "--tb=short"
        ]
        
        if debug:
            print(f"      [DEBUG] Running: {' '.join(cmd)}")
        
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
                print(f"      [DEBUG] pytest return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"      [DEBUG] Output: {output[-500:]}")
            
            # Check for errors
            if "error" in output.lower() and result.returncode != 0:
                # Extract error message
                error_match = re.search(r'(Error|Exception|ImportError|SyntaxError)[^\n]+', output)
                error_msg = error_match.group(0) if error_match else output[-200:]
                return None, error_msg
            
            # Parse coverage
            coverage_pct = None
            
            # Try JSON first
            coverage_json = test_dir / "coverage.json"
            if coverage_json.exists():
                try:
                    with open(coverage_json, encoding='utf-8') as f:
                        cov_data = json.load(f)
                        totals = cov_data.get("totals", {})
                        coverage_pct = totals.get("percent_covered", 0.0)
                    coverage_json.unlink()
                except Exception as e:
                    if debug:
                        print(f"      [DEBUG] Error parsing coverage.json: {e}")
            
            # Fallback to terminal output
            if coverage_pct is None:
                cov_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', output)
                if cov_match:
                    coverage_pct = float(cov_match.group(1))
            
            # Cleanup
            dotcoverage = test_dir / ".coverage"
            if dotcoverage.exists():
                dotcoverage.unlink()
            
            return coverage_pct if coverage_pct is not None else 0.0, None
            
        except subprocess.TimeoutExpired:
            return None, "Test timed out"
        except Exception as e:
            return None, str(e)
    
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
            content = test_file.read_text(encoding='utf-8', errors='replace')
            
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
        # Also fix relative imports to work without package structure
        for src_file in source_files:
            content = src_file.read_text(encoding='utf-8', errors='replace')
            
            # Fix relative imports: from .module import -> from module import
            # Also handle: from ._module import -> from _module import
            content = re.sub(r'from\s+\.(\w+)\s+import', r'from \1 import', content)
            
            # Fix relative imports with submodules: from .pkg.module import -> from pkg.module import  
            content = re.sub(r'from\s+\.(\w+\.\w+)\s+import', r'from \1 import', content)
            
            # Fix import . statements: import .module -> import module (rare but possible)
            content = re.sub(r'^import\s+\.(\w+)', r'import \1', content, flags=re.MULTILINE)
            
            dest_file = test_dir / src_file.name
            dest_file.write_text(content, encoding='utf-8')
            
            if debug:
                print(f"    [DEBUG] Copied {src_file.name} to test dir")
        
        # Fix imports in test files - replace "from src.X import" with "from X import"
        # Also fix any relative imports the AI might have generated
        for test_file in test_dir.glob("test_*.py"):
            test_content = test_file.read_text(encoding='utf-8', errors='replace')
            original = test_content
            
            # Fix src.module imports
            test_content = re.sub(r'from\s+src\.(\w+)\s+import', r'from \1 import', test_content)
            
            # Fix module.submodule imports that reference package structure
            # e.g., from manipulation.errors import -> from errors import
            test_content = re.sub(r'from\s+(\w+)\.errors\s+import', r'from errors import', test_content)
            test_content = re.sub(r'from\s+(\w+)\._regex\s+import', r'from _regex import', test_content)
            
            if test_content != original:
                test_file.write_text(test_content, encoding='utf-8')
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
                content = test_files[0].read_text(encoding='utf-8', errors='replace')
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
                "--cov-report=json:coverage.json",  # Explicit output file
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
            failed_count = int(failed.group(1)) if failed else 0
            error_count = int(error.group(1)) if error else 0
            measurement.tests_failed = failed_count + error_count
            
            # Parse coverage from terminal output
            # pytest-cov terminal output format:
            # Name                 Stmts   Miss  Cover
            # ----------------------------------------
            # errors.py               10      0   100%
            # manipulation.py        200     10    95%
            # TOTAL                   210     10    95%
            
            # First, try to get per-file coverage from terminal output
            # Pattern: filename.py    Stmts   Miss   Cover%
            file_cov_pattern = re.compile(r'^(\w+\.py)\s+\d+\s+\d+\s+(\d+)%', re.MULTILINE)
            for match in file_cov_pattern.finditer(output):
                filename = match.group(1)
                file_coverage = float(match.group(2))
                if not filename.startswith("test_"):
                    measurement.per_file_coverage[filename] = file_coverage
                    if debug:
                        print(f"    [DEBUG] Per-file from terminal: {filename} = {file_coverage}%")
            
            # Get total coverage from terminal
            cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            if cov_match:
                measurement.line_coverage = float(cov_match.group(1))
                if debug:
                    print(f"    [DEBUG] Coverage from terminal: {measurement.line_coverage}%")
            
            # Try to parse coverage.json for more detailed data (will override terminal if successful)
            coverage_file = test_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, encoding='utf-8') as f:
                    cov_data = json.load(f)
                    
                    if debug:
                        print(f"    [DEBUG] coverage.json keys: {list(cov_data.keys())}")
                    
                    totals = cov_data.get("totals", {})
                    if debug:
                        print(f"    [DEBUG] coverage.json totals: {totals}")
                    if totals.get("percent_covered"):
                        measurement.line_coverage = totals.get("percent_covered", 0.0)
                    if totals.get("percent_covered_branches"):
                        measurement.branch_coverage = totals.get("percent_covered_branches", 0.0)
                    
                    # Extract per-file coverage
                    # pytest-cov format: {"files": {"path/to/file.py": {"summary": {"percent_covered": X}}}}
                    files = cov_data.get("files", {})
                    if debug:
                        print(f"    [DEBUG] coverage.json has {len(files)} files")
                        if files:
                            first_key = list(files.keys())[0]
                            print(f"    [DEBUG] First file key: {first_key}")
                            print(f"    [DEBUG] First file data: {files[first_key]}")
                    
                    for filepath, file_data in files.items():
                        filename = Path(filepath).name
                        # Only include source files, not test files
                        if not filename.startswith("test_"):
                            summary = file_data.get("summary", {})
                            file_coverage = summary.get("percent_covered", 0.0)
                            measurement.per_file_coverage[filename] = round(file_coverage, 2)
                            if debug:
                                print(f"    [DEBUG] Added per-file: {filename} = {file_coverage}%")
                    
                    if debug:
                        print(f"    [DEBUG] Final per_file_coverage: {measurement.per_file_coverage}")
                
                # Cleanup coverage files
                coverage_file.unlink()
            else:
                if debug:
                    print(f"    [DEBUG] coverage.json not found at {coverage_file}")
            
            # Cleanup other coverage files
            dotcoverage = test_dir / ".coverage"
            if dotcoverage.exists():
                dotcoverage.unlink()
            htmlcov_dir = test_dir / "htmlcov"
            if htmlcov_dir.exists():
                shutil.rmtree(htmlcov_dir, ignore_errors=True)
                        
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
            content = tf.read_text(encoding='utf-8', errors='replace')
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
        generate_tests: Callable[[str, str], str],
        project_name: str = "unnamed",
        debug: bool = False
    ) -> CreationMeasurement:
        """Measure test creation for a Java project with JaCoCo coverage."""
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="java",
            project_name=project_name
        )
        
        project_path = Path(project_path).resolve()
        
        # Find Java source files
        source_files = list(project_path.glob("*.java"))
        source_files = [f for f in source_files if "Test" not in f.name]
        measurement.source_files_count = len(source_files)
        
        if not source_files:
            print("    Warning: No Java source files found")
            return measurement
        
        if debug:
            print(f"    [DEBUG] Found {len(source_files)} Java source files")
        
        # Create test output directory - clean it first to remove files from previous projects
        test_output_dir = self.work_dir / "generated_tests"
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Maven project structure
        self._setup_maven_project(test_output_dir, debug)
        
        # Maven source directories
        main_java = test_output_dir / "src" / "main" / "java"
        test_java = test_output_dir / "src" / "test" / "java"
        main_java.mkdir(parents=True, exist_ok=True)
        test_java.mkdir(parents=True, exist_ok=True)
        
        # Copy source files to src/main/java
        for source_file in source_files:
            shutil.copy(source_file, main_java / source_file.name)
        
        # Create a map of class name to file content for dependency resolution
        class_contents = {}
        for sf in source_files:
            class_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text(encoding='utf-8', errors='replace')
            # Pass the class name (filename without extension) as module_name
            module_name = source_file.stem  # e.g., "Slugify" from "Slugify.java"
            
            # Find what classes this file references from the project
            # Look for: new ClassName, ClassName.method, extends ClassName, implements ClassName
            referenced_classes = set()
            for class_name in class_contents.keys():
                if class_name != module_name:
                    # Check if this class is referenced in the source
                    if re.search(rf'\b{class_name}\b', source_code):
                        referenced_classes.add(class_name)
            
            # Build dependency code context
            dependency_code = ""
            for dep_class in referenced_classes:
                dep_content = class_contents[dep_class]
                # Truncate very long dependencies
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n// ... (truncated)"
                dependency_code += f"\n\n// === DEPENDENCY: {dep_class}.java ===\n{dep_content}"
            
            # Build the full context
            code_with_context = source_code
            if dependency_code:
                code_with_context = source_code + "\n\n// === PROJECT DEPENDENCIES (for reference) ===" + dependency_code
            elif len(source_files) > 1:
                other_files = [sf.stem for sf in source_files if sf != source_file]
                if other_files:
                    code_with_context = source_code + f"\n\n// NOTE: This project also contains these classes: {', '.join(other_files)}"
            
            try:
                generated_test = generate_tests(code_with_context, module_name)
                
                if generated_test:
                    # Extract code from markdown if present
                    generated_test = self._extract_code_from_response(generated_test, "java")
                    
                    test_filename = f"{source_file.stem}Test.java"
                    test_path = test_java / test_filename
                    test_path.write_text(generated_test, encoding='utf-8')
                    measurement.test_files_created += 1
                    print(f"    Generated: {test_filename}")
                    
            except Exception as e:
                print(f"    Error generating test for {source_file.name}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Analyze generated tests
        test_java = test_output_dir / "src" / "test" / "java"
        if test_java.exists():
            self._analyze_java_tests(measurement, test_java)
        
        # Run coverage measurement
        if measurement.test_files_created > 0:
            self._run_java_coverage(measurement, project_path, test_output_dir, debug)
        
        return measurement
    
    def _setup_maven_project(self, project_dir: Path, debug: bool = False):
        """Setup Maven project structure with pom.xml."""
        pom_file = project_dir / "pom.xml"
        
        if pom_file.exists():
            return  # Already setup
        
        if debug:
            print("    [DEBUG] Setting up Maven project...")
        
        pom_content = '''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.benchmark</groupId>
    <artifactId>test-project</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <maven.compiler.source>17</maven.compiler.source>
        <maven.compiler.target>17</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
        <lombok.version>1.18.36</lombok.version>
    </properties>

    <dependencies>
        <!-- JUnit 5 -->
        <dependency>
            <groupId>org.junit.jupiter</groupId>
            <artifactId>junit-jupiter</artifactId>
            <version>5.10.0</version>
            <scope>test</scope>
        </dependency>
        
        <!-- Lombok - using edge version for newer Java support -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <version>${lombok.version}</version>
            <scope>provided</scope>
        </dependency>
        
        <!-- ICU4J for Transliterator -->
        <dependency>
            <groupId>com.ibm.icu</groupId>
            <artifactId>icu4j</artifactId>
            <version>74.2</version>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Compiler plugin -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.13.0</version>
                <configuration>
                    <source>17</source>
                    <target>17</target>
                    <release>17</release>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                            <version>${lombok.version}</version>
                        </path>
                    </annotationProcessorPaths>
                </configuration>
            </plugin>
            
            <!-- Surefire plugin for running tests -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.2.2</version>
                <configuration>
                    <testFailureIgnore>true</testFailureIgnore>
                </configuration>
            </plugin>
            
            <!-- JaCoCo plugin for coverage -->
            <plugin>
                <groupId>org.jacoco</groupId>
                <artifactId>jacoco-maven-plugin</artifactId>
                <version>0.8.12</version>
                <executions>
                    <execution>
                        <goals>
                            <goal>prepare-agent</goal>
                        </goals>
                    </execution>
                    <execution>
                        <id>report</id>
                        <phase>test</phase>
                        <goals>
                            <goal>report</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>
</project>
'''
        pom_file.write_text(pom_content, encoding='utf-8')
    
    def _run_java_coverage(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Run Java tests with Maven and JaCoCo coverage."""
        
        import platform
        is_windows = platform.system() == "Windows"
        mvn_cmd = "mvn.cmd" if is_windows else "mvn"
        
        # Check if Maven is available
        try:
            subprocess.run([mvn_cmd, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    Warning: Maven not found, skipping Java coverage")
            return
        
        if debug:
            print("    [DEBUG] Running Maven test with coverage...")
        
        # Run mvn test (this compiles, runs tests, and generates coverage)
        try:
            test_result = subprocess.run(
                [mvn_cmd, "test", "-q"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=300,  # 5 minute timeout for Maven
                encoding='utf-8',
                errors='replace'
            )
            
            output = (test_result.stdout or "") + (test_result.stderr or "")
            
            if debug:
                print(f"    [DEBUG] Maven output (last 800 chars): {output[-800:]}")
            
            # Check for compilation errors
            if "COMPILATION ERROR" in output or "cannot find symbol" in output:
                if debug:
                    print("    [DEBUG] Compilation failed")
                measurement.compilation_errors.append(output[-1000:])
                return
            
            measurement.tests_compilable = measurement.test_files_created
            
            # Parse test results from Maven output
            # Tests run: 5, Failures: 1, Errors: 0, Skipped: 0
            tests_match = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", output)
            if tests_match:
                total = int(tests_match.group(1))
                failures = int(tests_match.group(2))
                errors = int(tests_match.group(3))
                measurement.tests_passing = total - failures - errors
                measurement.tests_failed = failures + errors
            
            # Parse coverage from JaCoCo CSV report
            coverage_csv = test_dir / "target" / "site" / "jacoco" / "jacoco.csv"
            if coverage_csv.exists():
                self._parse_jacoco_csv(measurement, coverage_csv, debug)
            else:
                # Try XML report
                coverage_xml = test_dir / "target" / "site" / "jacoco" / "jacoco.xml"
                if coverage_xml.exists():
                    self._parse_jacoco_xml(measurement, coverage_xml, debug)
                elif debug:
                    print("    [DEBUG] No JaCoCo report found")
                    
        except subprocess.TimeoutExpired:
            print("    Warning: Maven timed out")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Maven error: {e}")
    
    def _parse_jacoco_csv(self, measurement: CreationMeasurement, csv_path: Path, debug: bool = False):
        """Parse JaCoCo CSV coverage report."""
        try:
            coverage_data = csv_path.read_text(encoding='utf-8', errors='replace')
            lines = coverage_data.strip().split('\n')
            
            if debug:
                print(f"    [DEBUG] JaCoCo CSV has {len(lines)} lines")
                if len(lines) > 0:
                    print(f"    [DEBUG] Header: {lines[0]}")
                if len(lines) > 1:
                    print(f"    [DEBUG] First data row: {lines[1]}")
            
            if len(lines) > 1:
                total_covered = 0
                total_missed = 0
                for line in lines[1:]:  # Skip header
                    parts = line.split(',')
                    if debug:
                        print(f"    [DEBUG] Row has {len(parts)} columns, class={parts[2] if len(parts) > 2 else 'N/A'}")
                    if len(parts) >= 9:  # Need at least 9 columns (0-8)
                        try:
                            class_name = parts[2]  # CLASS column
                            missed = int(parts[7])  # LINE_MISSED
                            covered = int(parts[8])  # LINE_COVERED
                            total_missed += missed
                            total_covered += covered
                            
                            # Per-file coverage
                            file_total = missed + covered
                            if file_total > 0:
                                file_coverage = (covered / file_total) * 100
                                measurement.per_file_coverage[f"{class_name}.java"] = round(file_coverage, 2)
                            
                            if debug:
                                print(f"    [DEBUG] Class {class_name}: missed={missed}, covered={covered}")
                        except (ValueError, IndexError) as e:
                            if debug:
                                print(f"    [DEBUG] Parse error: {e}")
                
                total = total_covered + total_missed
                if total > 0:
                    measurement.line_coverage = (total_covered / total) * 100
                    if debug:
                        print(f"    [DEBUG] Total coverage: {total_covered}/{total} = {measurement.line_coverage:.1f}%")
                        if measurement.per_file_coverage:
                            print(f"    [DEBUG] Per-file coverage: {measurement.per_file_coverage}")
                elif debug:
                    print(f"    [DEBUG] No coverage data found (total=0)")
            
            # Cleanup coverage files
            csv_path.unlink()
            
            # Also cleanup jacoco.exec
            jacoco_exec = csv_path.parent.parent.parent / "jacoco.exec"
            if jacoco_exec.exists():
                jacoco_exec.unlink()
                
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Error parsing coverage CSV: {e}")
    
    def _parse_jacoco_xml(self, measurement: CreationMeasurement, xml_path: Path, debug: bool = False):
        """Parse JaCoCo XML coverage report."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Find LINE counter
            for counter in root.findall(".//counter[@type='LINE']"):
                missed = int(counter.get('missed', 0))
                covered = int(counter.get('covered', 0))
                total = missed + covered
                if total > 0:
                    measurement.line_coverage = (covered / total) * 100
                    if debug:
                        print(f"    [DEBUG] Coverage: {measurement.line_coverage:.1f}%")
                    break
            
            # Cleanup coverage files
            xml_path.unlink()
            
            # Also cleanup jacoco.exec
            jacoco_exec = xml_path.parent.parent.parent / "jacoco.exec"
            if jacoco_exec.exists():
                jacoco_exec.unlink()
                
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Error parsing coverage XML: {e}")
    
    def _download_file(self, url: str, dest: Path):
        """Download a file from URL."""
        import urllib.request
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"    Warning: Could not download {url}: {e}")
    
    def _analyze_java_tests(
        self,
        measurement: CreationMeasurement,
        test_dir: Path
    ):
        """Analyze generated Java tests."""
        test_files = list(test_dir.glob("*Test.java"))
        measurement.tests_generated = len(test_files)
        
        for test_file in test_files:
            content = test_file.read_text(encoding='utf-8', errors='replace')
            
            # Check for basic syntax (simplified)
            if "class" in content and "@Test" in content:
                measurement.tests_compilable += 1
    
    def measure_javascript(
        self,
        bot_name: str,
        project_path: str,
        generate_tests: Callable[[str, str], str],
        project_name: str = "unnamed",
        debug: bool = False
    ) -> CreationMeasurement:
        """Measure test creation for a JavaScript project with Jest coverage."""
        measurement = CreationMeasurement(
            bot_name=bot_name,
            language="javascript",
            project_name=project_name
        )
        
        project_path = Path(project_path).resolve()
        
        # Find JS source files
        source_files = list(project_path.glob("*.js"))
        source_files.extend(project_path.glob("*.ts"))
        source_files = [f for f in source_files if ".test." not in f.name and ".spec." not in f.name]
        measurement.source_files_count = len(source_files)
        
        if not source_files:
            print("    Warning: No JavaScript source files found")
            return measurement
        
        if debug:
            print(f"    [DEBUG] Found {len(source_files)} JavaScript source files")
        
        # Create test output directory - clean it first to remove files from previous projects
        test_output_dir = self.work_dir / "generated_tests"
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a map of module name to file content for dependency resolution
        module_contents = {}
        for sf in source_files:
            module_contents[sf.stem] = sf.read_text(encoding='utf-8', errors='replace')
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text(encoding='utf-8', errors='replace')
            # Pass the filename (without extension) as module_name for correct imports
            module_name = source_file.stem  # e.g., "camelCase" from "camelCase.js"
            
            # Find what modules this file requires from the project
            # Look for: require('./module') or import from './module'
            required_modules = set()
            for mod_name in module_contents.keys():
                if mod_name != module_name:
                    # Check for require or import of this module
                    if re.search(rf"(?:require\s*\(\s*['\"]\./{mod_name}|from\s+['\"]\./{mod_name})", source_code):
                        required_modules.add(mod_name)
            
            # Build dependency code context
            dependency_code = ""
            for dep_mod in required_modules:
                dep_content = module_contents[dep_mod]
                # Truncate very long dependencies
                if len(dep_content) > 2000:
                    dep_content = dep_content[:2000] + "\n// ... (truncated)"
                dependency_code += f"\n\n// === DEPENDENCY: {dep_mod}.js ===\n{dep_content}"
            
            # Build the full context
            code_with_context = source_code
            if dependency_code:
                code_with_context = source_code + "\n\n// === PROJECT DEPENDENCIES (for reference) ===" + dependency_code
            elif len(source_files) > 1:
                other_files = [sf.stem for sf in source_files if sf != source_file]
                if other_files:
                    code_with_context = source_code + f"\n\n// NOTE: This project also contains these modules: {', '.join(other_files)}"
            
            try:
                generated_test = generate_tests(code_with_context, module_name)
                
                if generated_test:
                    # Extract code from markdown if present
                    generated_test = self._extract_code_from_response(generated_test, "javascript")
                    
                    test_filename = f"{source_file.stem}.test.js"
                    test_path = test_output_dir / test_filename
                    test_path.write_text(generated_test, encoding='utf-8')
                    measurement.test_files_created += 1
                    print(f"    Generated: {test_filename}")
                    
            except Exception as e:
                print(f"    Error generating test for {source_file.name}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Analyze generated tests
        self._analyze_js_tests(measurement, test_output_dir)
        
        # Run coverage measurement
        if measurement.test_files_created > 0:
            self._run_javascript_coverage(measurement, project_path, test_output_dir, debug)
        
        return measurement
    
    def _fix_javascript_imports(self, test_file: Path, debug: bool = False):
        """Fix ES Module imports to CommonJS requires."""
        content = test_file.read_text(encoding='utf-8', errors='replace')
        original = content
        
        # Remove ES Module imports that Jest doesn't support
        # import { jest } from '@jest/globals';
        content = re.sub(r"import\s*\{\s*jest\s*\}\s*from\s*['\"]@jest/globals['\"];?\n?", "", content)
        
        # Convert ES Module imports to CommonJS requires
        # import { func1, func2 } from './module'; -> const { func1, func2 } = require('./module');
        content = re.sub(
            r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"];?",
            r"const {\1} = require('\2');",
            content
        )
        
        # import module from './module'; -> const module = require('./module');
        content = re.sub(
            r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"];?",
            r"const \1 = require('\2');",
            content
        )
        
        # import * as module from './module'; -> const module = require('./module');
        content = re.sub(
            r"import\s*\*\s*as\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"];?",
            r"const \1 = require('\2');",
            content
        )
        
        # Fix path patterns
        # require('./src/module') -> require('./module')
        content = re.sub(r"require\(['\"]\.\/src\/", "require('./", content)
        
        if content != original and debug:
            print(f"    [DEBUG] Fixed imports in {test_file.name}")
        
        test_file.write_text(content, encoding='utf-8')
    
    def _convert_esm_to_commonjs(self, js_file: Path, debug: bool = False):
        """Convert ES Module syntax to CommonJS in source files."""
        content = js_file.read_text(encoding='utf-8', errors='replace')
        original = content
        
        # Convert export default function/class/const
        # export default function name() -> function name() ... module.exports = name;
        # For simplicity, we'll convert to module.exports pattern
        
        # export default X -> module.exports = X
        content = re.sub(
            r'export\s+default\s+',
            'module.exports = ',
            content
        )
        
        # export function name() -> function name() ... then add to exports
        # First, find all exported function names
        exported_functions = re.findall(r'export\s+function\s+(\w+)', content)
        exported_consts = re.findall(r'export\s+(?:const|let|var)\s+(\w+)', content)
        exported_classes = re.findall(r'export\s+class\s+(\w+)', content)
        
        # Remove 'export' keyword from declarations
        content = re.sub(r'export\s+(function|class|const|let|var)\s+', r'\1 ', content)
        
        # Add module.exports at the end if there are named exports
        all_exports = exported_functions + exported_consts + exported_classes
        if all_exports:
            exports_obj = ', '.join(all_exports)
            # Check if module.exports already exists
            if 'module.exports' not in content:
                content = content.rstrip() + f'\n\nmodule.exports = {{ {exports_obj} }};\n'
        
        # Convert import statements to require
        # import { x, y } from './module' -> const { x, y } = require('./module')
        content = re.sub(
            r"import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"];?",
            r"const {\1} = require('\2');",
            content
        )
        
        # import x from './module' -> const x = require('./module')
        content = re.sub(
            r"import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"];?",
            r"const \1 = require('\2');",
            content
        )
        
        if content != original and debug:
            print(f"    [DEBUG] Converted {js_file.name} from ESM to CommonJS")
        
        js_file.write_text(content, encoding='utf-8')
    
    def _run_javascript_coverage(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Run JavaScript tests with Jest coverage."""
        
        # Copy source files to test directory and collect their names
        source_file_names = []
        for src_file in project_path.glob("*.js"):
            if ".test." not in src_file.name and ".spec." not in src_file.name:
                dest_file = test_dir / src_file.name
                shutil.copy(src_file, dest_file)
                # Convert source file from ES Modules to CommonJS
                self._convert_esm_to_commonjs(dest_file, debug)
                source_file_names.append(src_file.name)
        
        if debug:
            print(f"    [DEBUG] Source files for coverage: {source_file_names}")
        
        # Fix imports in test files (convert ES Modules to CommonJS)
        for test_file in test_dir.glob("*.test.js"):
            self._fix_javascript_imports(test_file, debug)
        for test_file in test_dir.glob("*.spec.js"):
            self._fix_javascript_imports(test_file, debug)
        
        # Check if node is available
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    Warning: Node.js not found, skipping coverage")
            return
        
        # Create package.json if not exists
        package_json = test_dir / "package.json"
        if not package_json.exists():
            package_json.write_text(json.dumps({
                "name": "test-project",
                "version": "1.0.0",
                "scripts": {
                    "test": "jest --coverage --coverageReporters=json-summary"
                },
                "devDependencies": {
                    "jest": "^29.0.0"
                }
            }, indent=2))
        
        # Build collectCoverageFrom array - only include actual source files
        coverage_patterns = [f"./{name}" for name in source_file_names]
        
        # Create jest.config.js with explicit source file coverage
        jest_config = test_dir / "jest.config.js"
        jest_config.write_text(f"""
module.exports = {{
    testEnvironment: 'node',
    collectCoverage: true,
    collectCoverageFrom: {json.dumps(coverage_patterns)},
    coverageReporters: ['json-summary', 'text'],
    coverageDirectory: './coverage',
    testMatch: ['**/*.test.js', '**/*.spec.js'],
    moduleFileExtensions: ['js', 'json'],
}};
""")
        
        # Determine correct command names for Windows vs Unix
        import platform
        is_windows = platform.system() == "Windows"
        npm_cmd = "npm.cmd" if is_windows else "npm"
        npx_cmd = "npx.cmd" if is_windows else "npx"
        
        # Install Jest if needed
        node_modules = test_dir / "node_modules"
        if not node_modules.exists():
            if debug:
                print("    [DEBUG] Installing Jest...")
            try:
                install_result = subprocess.run(
                    [npm_cmd, "install", "--save-dev", "jest"],
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=120,
                    shell=is_windows,  # Use shell on Windows for .cmd files
                    encoding='utf-8',
                    errors='replace'
                )
                if install_result.returncode != 0 and debug:
                    print(f"    [DEBUG] npm install warning: {(install_result.stderr or '')[:300]}")
            except FileNotFoundError:
                print("    Warning: npm not found. Please install Node.js.")
                return
        
        # Run Jest with coverage
        if debug:
            print("    [DEBUG] Running Jest with coverage...")
        
        try:
            test_result = subprocess.run(
                [npx_cmd, "jest", "--coverage", "--coverageReporters=json-summary", "--coverageReporters=text", "--no-colors"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=120,
                shell=is_windows,  # Use shell on Windows for .cmd files
                encoding='utf-8',
                errors='replace'  # Handle special characters in Jest output
            )
            
            output = (test_result.stdout or "") + (test_result.stderr or "")
            
            if debug:
                print(f"    [DEBUG] Jest output: {output[-800:]}")
            
            # Parse test results from output
            # Jest output: Tests: X passed, Y failed, Z total
            passed = re.search(r"Tests:\s+(\d+)\s+passed", output)
            failed = re.search(r"(\d+)\s+failed", output)
            
            measurement.tests_passing = int(passed.group(1)) if passed else 0
            measurement.tests_failed = int(failed.group(1)) if failed else 0
            measurement.tests_compilable = measurement.test_files_created
            
            # Parse coverage from JSON summary
            coverage_file = test_dir / "coverage" / "coverage-summary.json"
            if coverage_file.exists():
                with open(coverage_file, encoding='utf-8') as f:
                    coverage_data = json.load(f)
                    total = coverage_data.get("total", {})
                    lines = total.get("lines", {})
                    pct = lines.get("pct", 0)
                    
                    # Handle "Unknown" or other non-numeric values
                    if isinstance(pct, (int, float)):
                        measurement.line_coverage = float(pct)
                    else:
                        # pct is "Unknown" or some other string
                        measurement.line_coverage = 0.0
                    
                    # Extract per-file coverage
                    for filepath, file_data in coverage_data.items():
                        if filepath != "total":
                            filename = Path(filepath).name
                            # Only include source files, not test files
                            if not filename.endswith(".test.js") and not filename.endswith(".spec.js"):
                                file_lines = file_data.get("lines", {})
                                file_pct = file_lines.get("pct", 0)
                                if isinstance(file_pct, (int, float)):
                                    measurement.per_file_coverage[filename] = round(float(file_pct), 2)
                    
                    if debug:
                        print(f"    [DEBUG] Coverage: {measurement.line_coverage:.1f}%")
                        if measurement.per_file_coverage:
                            print(f"    [DEBUG] Per-file coverage: {measurement.per_file_coverage}")
                
                # Cleanup coverage directory
                coverage_dir = test_dir / "coverage"
                if coverage_dir.exists():
                    shutil.rmtree(coverage_dir, ignore_errors=True)
            else:
                # Try to parse from text output
                # All files |   85.71 |      100 |     100 |   85.71
                cov_match = re.search(r"All files\s*\|\s*([\d.]+)", output)
                if cov_match:
                    measurement.line_coverage = float(cov_match.group(1))
                    
        except subprocess.TimeoutExpired:
            print("    Warning: Jest timed out")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Jest error: {e}")
    
    def _analyze_js_tests(self, measurement: CreationMeasurement, test_dir: Path):
        """Analyze generated JavaScript tests."""
        test_files = list(test_dir.glob("*.test.js"))
        test_files.extend(test_dir.glob("*.spec.js"))
        measurement.tests_generated = len(test_files)
        
        for test_file in test_files:
            # Check syntax with Node
            result = subprocess.run(
                ["node", "--check", str(test_file)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                measurement.tests_compilable += 1
            else:
                measurement.compilation_errors.append({
                    "file": test_file.name,
                    "error": result.stderr or ""
                })

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