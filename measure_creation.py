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
        
        # Create test output directory
        test_output_dir = self.work_dir / "generated_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text(encoding='utf-8')
            
            try:
                generated_test = generate_tests(source_code)
                
                if generated_test:
                    # Extract code from markdown if present
                    generated_test = self._extract_code_from_response(generated_test, "java")
                    
                    test_filename = f"{source_file.stem}Test.java"
                    test_path = test_output_dir / test_filename
                    test_path.write_text(generated_test, encoding='utf-8')
                    measurement.test_files_created += 1
                    print(f"    Generated: {test_filename}")
                    
            except Exception as e:
                print(f"    Error generating test for {source_file.name}: {e}")
        
        measurement.generation_time_seconds = time.time() - start_time
        
        # Run coverage measurement
        if measurement.test_files_created > 0:
            self._run_java_coverage(measurement, project_path, test_output_dir, debug)
        
        return measurement
    
    def _run_java_coverage(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Run Java tests with JaCoCo coverage."""
        
        # Copy source files to test directory
        for src_file in project_path.glob("*.java"):
            if "Test" not in src_file.name:
                shutil.copy(src_file, test_dir / src_file.name)
        
        # Check if javac and java are available
        try:
            subprocess.run(["javac", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("    Warning: javac not found, skipping coverage")
            return
        
        # Download JUnit and JaCoCo if not present
        lib_dir = test_dir / "lib"
        lib_dir.mkdir(exist_ok=True)
        
        junit_jar = lib_dir / "junit-platform-console-standalone.jar"
        jacoco_agent = lib_dir / "jacocoagent.jar"
        jacoco_cli = lib_dir / "jacococli.jar"
        
        # Download dependencies if needed
        if not junit_jar.exists():
            if debug:
                print("    [DEBUG] Downloading JUnit...")
            self._download_file(
                "https://repo1.maven.org/maven2/org/junit/platform/junit-platform-console-standalone/1.10.0/junit-platform-console-standalone-1.10.0.jar",
                junit_jar
            )
        
        if not jacoco_agent.exists():
            if debug:
                print("    [DEBUG] Downloading JaCoCo agent...")
            self._download_file(
                "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.agent/0.8.11/org.jacoco.agent-0.8.11-runtime.jar",
                jacoco_agent
            )
        
        if not jacoco_cli.exists():
            if debug:
                print("    [DEBUG] Downloading JaCoCo CLI...")
            self._download_file(
                "https://repo1.maven.org/maven2/org/jacoco/org.jacoco.cli/0.8.11/org.jacoco.cli-0.8.11-nodeps.jar",
                jacoco_cli
            )
        
        # Compile source and test files
        java_files = list(test_dir.glob("*.java"))
        if not java_files:
            return
        
        classpath = f"{junit_jar}{os.pathsep}."
        
        if debug:
            print(f"    [DEBUG] Compiling {len(java_files)} Java files...")
        
        compile_result = subprocess.run(
            ["javac", "-cp", classpath, "-d", "."] + [str(f) for f in java_files],
            capture_output=True,
            text=True,
            cwd=str(test_dir)
        )
        
        if compile_result.returncode != 0:
            if debug:
                print(f"    [DEBUG] Compilation failed: {compile_result.stderr[:500]}")
            measurement.compilation_errors.append(compile_result.stderr)
            return
        
        measurement.tests_compilable = measurement.test_files_created
        
        # Find test classes
        test_classes = [f.stem for f in test_dir.glob("*Test.java")]
        if not test_classes:
            return
        
        # Run tests with JaCoCo agent
        jacoco_exec = test_dir / "jacoco.exec"
        
        if debug:
            print(f"    [DEBUG] Running tests with JaCoCo...")
        
        test_result = subprocess.run(
            [
                "java",
                f"-javaagent:{jacoco_agent}=destfile={jacoco_exec}",
                "-jar", str(junit_jar),
                "--class-path", ".",
                "--scan-class-path"
            ],
            capture_output=True,
            text=True,
            cwd=str(test_dir),
            timeout=120
        )
        
        if debug:
            print(f"    [DEBUG] Test output: {test_result.stdout[-500:]}")
        
        # Parse test results
        output = test_result.stdout
        passed = re.search(r"(\d+) tests successful", output)
        failed = re.search(r"(\d+) tests failed", output)
        
        measurement.tests_passing = int(passed.group(1)) if passed else 0
        
        # Generate coverage report
        if jacoco_exec.exists():
            # Get source class files (not test classes)
            source_classes = [f.stem for f in project_path.glob("*.java") if "Test" not in f.name]
            
            report_result = subprocess.run(
                [
                    "java", "-jar", str(jacoco_cli),
                    "report", str(jacoco_exec),
                    "--classfiles", ".",
                    "--sourcefiles", ".",
                    "--csv", "coverage.csv"
                ],
                capture_output=True,
                text=True,
                cwd=str(test_dir)
            )
            
            # Parse coverage CSV
            coverage_csv = test_dir / "coverage.csv"
            if coverage_csv.exists():
                coverage_data = coverage_csv.read_text()
                lines = coverage_data.strip().split('\n')
                if len(lines) > 1:
                    total_covered = 0
                    total_missed = 0
                    for line in lines[1:]:  # Skip header
                        parts = line.split(',')
                        if len(parts) >= 8:
                            # CSV format: GROUP,PACKAGE,CLASS,INSTRUCTION_MISSED,INSTRUCTION_COVERED,...
                            try:
                                missed = int(parts[7])  # LINE_MISSED
                                covered = int(parts[8])  # LINE_COVERED
                                total_missed += missed
                                total_covered += covered
                            except (ValueError, IndexError):
                                pass
                    
                    total = total_covered + total_missed
                    if total > 0:
                        measurement.line_coverage = (total_covered / total) * 100
                        if debug:
                            print(f"    [DEBUG] Coverage: {measurement.line_coverage:.1f}%")
    
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
        
        # Create test output directory
        test_output_dir = self.work_dir / "generated_tests"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate tests
        start_time = time.time()
        
        for source_file in source_files:
            source_code = source_file.read_text(encoding='utf-8')
            
            try:
                generated_test = generate_tests(source_code)
                
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
        
        # Run coverage measurement
        if measurement.test_files_created > 0:
            self._run_javascript_coverage(measurement, project_path, test_output_dir, debug)
        
        return measurement
    
    def _fix_javascript_imports(self, test_file: Path, debug: bool = False):
        """Fix ES Module imports to CommonJS requires."""
        content = test_file.read_text(encoding='utf-8')
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
    
    def _run_javascript_coverage(
        self,
        measurement: CreationMeasurement,
        project_path: Path,
        test_dir: Path,
        debug: bool = False
    ):
        """Run JavaScript tests with Jest coverage."""
        
        # Copy source files to test directory
        for src_file in project_path.glob("*.js"):
            if ".test." not in src_file.name and ".spec." not in src_file.name:
                shutil.copy(src_file, test_dir / src_file.name)
        
        # Fix imports in test files (convert ES Modules to CommonJS)
        for test_file in test_dir.glob("*.test.js"):
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
        
        # Create jest.config.js
        jest_config = test_dir / "jest.config.js"
        jest_config.write_text("""
module.exports = {
    testEnvironment: 'node',
    collectCoverage: true,
    coverageReporters: ['json-summary', 'text'],
    coverageDirectory: './coverage',
    testMatch: ['**/*.test.js', '**/*.spec.js'],
    moduleFileExtensions: ['js', 'json'],
};
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
                    
                    if debug:
                        print(f"    [DEBUG] Coverage: {measurement.line_coverage:.1f}%")
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