#!/usr/bin/env python3
"""
AI-Powered Testing Bot Benchmarking Framework
University of Calgary - CPSC 599

This framework evaluates AI testing bots across three categories:
1. Test Creation - Measuring coverage, correctness, and generation speed
2. Test Execution - Measuring test suite quality and execution metrics
3. Test Maintenance - Measuring error detection and repair capabilities

Supports: Python, Java, JavaScript
"""

import subprocess
import time
import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime


# ============================================================================
# DATA CLASSES FOR METRICS
# ============================================================================

@dataclass
class TestCreationMetrics:
    """Metrics for evaluating test creation capabilities."""
    # Coverage Metrics
    line_coverage: float = 0.0          # Percentage of lines covered
    branch_coverage: float = 0.0        # Percentage of branches covered
    statement_coverage: float = 0.0     # Percentage of statements covered
    function_coverage: float = 0.0      # Percentage of functions covered
    
    # Quality Metrics
    tests_generated: int = 0            # Total number of tests generated
    tests_compilable: int = 0           # Tests that compile without errors
    tests_passing: int = 0              # Tests that pass on first run
    tests_with_assertions: int = 0      # Tests containing meaningful assertions
    
    # Efficiency Metrics
    generation_time_seconds: float = 0.0  # Time to generate all tests
    tokens_used: int = 0                  # API tokens consumed (if applicable)
    
    # Derived Metrics (calculated)
    compilation_rate: float = 0.0       # tests_compilable / tests_generated
    pass_rate: float = 0.0              # tests_passing / tests_compilable
    assertion_density: float = 0.0      # tests_with_assertions / tests_generated
    
    def calculate_derived_metrics(self):
        if self.tests_generated > 0:
            self.compilation_rate = self.tests_compilable / self.tests_generated
            self.assertion_density = self.tests_with_assertions / self.tests_generated
        if self.tests_compilable > 0:
            self.pass_rate = self.tests_passing / self.tests_compilable


@dataclass
class TestExecutionMetrics:
    """Metrics for evaluating test suite execution capabilities."""
    # Suite Organization
    total_test_files: int = 0           # Number of test files created
    total_test_cases: int = 0           # Total test cases in suite
    test_suite_size_kb: float = 0.0     # Size of generated test suite
    
    # Execution Performance
    execution_time_seconds: float = 0.0  # Total time to run test suite
    avg_test_time_ms: float = 0.0       # Average time per test
    parallel_execution_supported: bool = False
    
    # Fault Detection
    bugs_detected: int = 0              # Number of known bugs found
    false_positives: int = 0            # Tests failing incorrectly
    false_negatives: int = 0            # Missed bugs that should be caught
    mutation_score: float = 0.0         # Percentage of mutants killed
    
    # Suite Quality
    test_isolation_score: float = 0.0   # Tests that run independently (0-1)
    flaky_test_count: int = 0           # Tests with inconsistent results
    redundant_test_count: int = 0       # Duplicate or unnecessary tests
    
    # Derived Metrics
    fault_detection_rate: float = 0.0   # bugs_detected / total_known_bugs
    precision: float = 0.0              # true_positives / (true_positives + false_positives)
    recall: float = 0.0                 # true_positives / (true_positives + false_negatives)
    
    def calculate_derived_metrics(self, total_known_bugs: int):
        if total_known_bugs > 0:
            self.fault_detection_rate = self.bugs_detected / total_known_bugs
        true_positives = self.bugs_detected
        if (true_positives + self.false_positives) > 0:
            self.precision = true_positives / (true_positives + self.false_positives)
        if (true_positives + self.false_negatives) > 0:
            self.recall = true_positives / (true_positives + self.false_negatives)


@dataclass
class TestMaintenanceMetrics:
    """Metrics for evaluating test maintenance/repair capabilities."""
    # Error Detection
    total_broken_tests: int = 0         # Tests that need fixing
    errors_detected: int = 0            # Errors identified by the bot
    error_detection_rate: float = 0.0   # errors_detected / total_broken_tests
    
    # Repair Success
    repair_attempts: int = 0            # Number of repair attempts made
    successful_repairs: int = 0         # Tests successfully fixed
    partial_repairs: int = 0            # Tests partially fixed
    failed_repairs: int = 0             # Repair attempts that failed
    
    # Repair Quality
    repair_correctness: float = 0.0     # Repairs that don't break other tests
    regression_introduced: int = 0      # New failures caused by repairs
    
    # Efficiency
    repair_time_seconds: float = 0.0    # Time to complete all repairs
    avg_repair_time_seconds: float = 0.0  # Average time per repair
    human_intervention_needed: int = 0   # Repairs requiring human help
    
    # Derived Metrics
    repair_success_rate: float = 0.0    # successful_repairs / total_broken_tests
    automation_rate: float = 0.0        # 1 - (human_intervention / repair_attempts)
    
    def calculate_derived_metrics(self):
        if self.total_broken_tests > 0:
            self.error_detection_rate = self.errors_detected / self.total_broken_tests
            self.repair_success_rate = self.successful_repairs / self.total_broken_tests
        if self.repair_attempts > 0:
            self.avg_repair_time_seconds = self.repair_time_seconds / self.repair_attempts
            self.automation_rate = 1 - (self.human_intervention_needed / self.repair_attempts)


@dataclass
class UsabilityMetrics:
    """Metrics for evaluating user experience and ease of use."""
    setup_time_minutes: float = 0.0     # Time to install and configure
    learning_curve_rating: int = 0      # 1-5 scale (1=hard, 5=easy)
    documentation_quality: int = 0      # 1-5 scale
    integration_ease: int = 0           # 1-5 scale (IDE, CI/CD integration)
    output_clarity: int = 0             # 1-5 scale (readability of generated tests)
    error_message_quality: int = 0      # 1-5 scale
    customization_options: int = 0      # 1-5 scale


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single bot."""
    bot_name: str
    language: str  # python, java, javascript
    project_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    creation_metrics: TestCreationMetrics = field(default_factory=TestCreationMetrics)
    execution_metrics: TestExecutionMetrics = field(default_factory=TestExecutionMetrics)
    maintenance_metrics: TestMaintenanceMetrics = field(default_factory=TestMaintenanceMetrics)
    usability_metrics: UsabilityMetrics = field(default_factory=UsabilityMetrics)
    
    # Overall Scores (0-100)
    creation_score: float = 0.0
    execution_score: float = 0.0
    maintenance_score: float = 0.0
    usability_score: float = 0.0
    overall_score: float = 0.0
    
    def calculate_scores(self, total_known_bugs: int = 0):
        """Calculate composite scores for each category."""
        self.creation_metrics.calculate_derived_metrics()
        self.execution_metrics.calculate_derived_metrics(total_known_bugs)
        self.maintenance_metrics.calculate_derived_metrics()
        
        # Creation Score (weighted average)
        self.creation_score = (
            self.creation_metrics.line_coverage * 25 +
            self.creation_metrics.branch_coverage * 25 +
            self.creation_metrics.compilation_rate * 100 * 20 +
            self.creation_metrics.pass_rate * 100 * 20 +
            self.creation_metrics.assertion_density * 100 * 10
        )
        
        # Execution Score
        self.execution_score = (
            self.execution_metrics.fault_detection_rate * 100 * 30 +
            self.execution_metrics.precision * 100 * 25 +
            self.execution_metrics.recall * 100 * 25 +
            (1 - min(self.execution_metrics.flaky_test_count / max(self.execution_metrics.total_test_cases, 1), 1)) * 100 * 20
        )
        
        # Maintenance Score
        self.maintenance_score = (
            self.maintenance_metrics.repair_success_rate * 100 * 40 +
            self.maintenance_metrics.automation_rate * 100 * 30 +
            self.maintenance_metrics.error_detection_rate * 100 * 20 +
            (1 - min(self.maintenance_metrics.regression_introduced / max(self.maintenance_metrics.successful_repairs, 1), 1)) * 100 * 10
        )
        
        # Usability Score
        usability = self.usability_metrics
        self.usability_score = (
            (usability.learning_curve_rating / 5) * 100 * 20 +
            (usability.documentation_quality / 5) * 100 * 20 +
            (usability.integration_ease / 5) * 100 * 20 +
            (usability.output_clarity / 5) * 100 * 20 +
            (usability.error_message_quality / 5) * 100 * 10 +
            (usability.customization_options / 5) * 100 * 10
        )
        
        # Overall Score (weighted by category importance)
        self.overall_score = (
            self.creation_score * 0.35 +
            self.execution_score * 0.30 +
            self.maintenance_score * 0.20 +
            self.usability_score * 0.15
        )
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# LANGUAGE-SPECIFIC ANALYZERS
# ============================================================================

class CoverageAnalyzer(ABC):
    """Abstract base class for language-specific coverage analysis."""
    
    @abstractmethod
    def run_coverage(self, project_path: str, test_path: str) -> dict:
        """Run coverage tool and return metrics."""
        pass
    
    @abstractmethod
    def count_assertions(self, test_file: str) -> int:
        """Count assertions in a test file."""
        pass
    
    @abstractmethod
    def check_compilation(self, test_path: str) -> tuple[int, int]:
        """Check if tests compile. Returns (total, compilable)."""
        pass


class PythonAnalyzer(CoverageAnalyzer):
    """Coverage analyzer for Python projects using pytest-cov."""
    
    def run_coverage(self, project_path: str, test_path: str) -> dict:
        """Run pytest with coverage and return metrics."""
        cmd = [
            "python", "-m", "pytest", test_path,
            f"--cov={project_path}",
            "--cov-report=json:coverage.json",
            "--cov-report=term",
            "-v", "--tb=short"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_path)
        execution_time = time.time() - start_time
        
        metrics = {
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "execution_time": execution_time,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_total": 0
        }
        
        # Parse coverage.json if exists
        coverage_file = Path(project_path) / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                cov_data = json.load(f)
                if "totals" in cov_data:
                    metrics["line_coverage"] = cov_data["totals"].get("percent_covered", 0)
                    metrics["branch_coverage"] = cov_data["totals"].get("percent_covered_branches", 0)
        
        # Parse test results from output
        output = result.stdout + result.stderr
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        
        if passed_match:
            metrics["tests_passed"] = int(passed_match.group(1))
        if failed_match:
            metrics["tests_failed"] = int(failed_match.group(1))
        metrics["tests_total"] = metrics["tests_passed"] + metrics["tests_failed"]
        
        return metrics
    
    def count_assertions(self, test_file: str) -> int:
        """Count assert statements in Python test file."""
        count = 0
        assertion_patterns = [
            r'\bassert\s+',
            r'\.assert\w+\(',
            r'self\.assert\w+\(',
            r'pytest\.raises\(',
        ]
        
        with open(test_file, 'r') as f:
            content = f.read()
            for pattern in assertion_patterns:
                count += len(re.findall(pattern, content))
        return count
    
    def check_compilation(self, test_path: str) -> tuple[int, int]:
        """Check Python test files for syntax errors."""
        import py_compile
        test_files = list(Path(test_path).rglob("test_*.py"))
        total = len(test_files)
        compilable = 0
        
        for test_file in test_files:
            try:
                py_compile.compile(str(test_file), doraise=True)
                compilable += 1
            except py_compile.PyCompileError:
                pass
        
        return total, compilable


class JavaAnalyzer(CoverageAnalyzer):
    """Coverage analyzer for Java projects using JaCoCo."""
    
    def run_coverage(self, project_path: str, test_path: str) -> dict:
        """Run Maven with JaCoCo coverage."""
        cmd = ["mvn", "test", "jacoco:report", "-q"]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_path)
        execution_time = time.time() - start_time
        
        metrics = {
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "execution_time": execution_time,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_total": 0
        }
        
        # Parse JaCoCo report (typically in target/site/jacoco/jacoco.xml)
        jacoco_report = Path(project_path) / "target/site/jacoco/jacoco.xml"
        if jacoco_report.exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(jacoco_report)
            root = tree.getroot()
            
            for counter in root.findall(".//counter"):
                ctype = counter.get("type")
                missed = int(counter.get("missed", 0))
                covered = int(counter.get("covered", 0))
                total = missed + covered
                
                if total > 0:
                    if ctype == "LINE":
                        metrics["line_coverage"] = (covered / total) * 100
                    elif ctype == "BRANCH":
                        metrics["branch_coverage"] = (covered / total) * 100
        
        return metrics
    
    def count_assertions(self, test_file: str) -> int:
        """Count assertions in Java test file."""
        count = 0
        assertion_patterns = [
            r'\bassert\w+\(',
            r'@Test',
            r'Assertions\.\w+\(',
            r'assertThat\(',
        ]
        
        with open(test_file, 'r') as f:
            content = f.read()
            for pattern in assertion_patterns:
                count += len(re.findall(pattern, content))
        return count
    
    def check_compilation(self, test_path: str) -> tuple[int, int]:
        """Check Java test compilation using javac."""
        test_files = list(Path(test_path).rglob("*Test.java"))
        total = len(test_files)
        
        # Try to compile all test files
        cmd = ["mvn", "test-compile", "-q"]
        result = subprocess.run(cmd, capture_output=True, cwd=str(Path(test_path).parent))
        
        if result.returncode == 0:
            return total, total
        return total, 0  # Simplified - in practice parse compile errors


class JavaScriptAnalyzer(CoverageAnalyzer):
    """Coverage analyzer for JavaScript projects using Jest/NYC."""
    
    def run_coverage(self, project_path: str, test_path: str) -> dict:
        """Run Jest with coverage."""
        cmd = ["npm", "test", "--", "--coverage", "--json", "--outputFile=test-results.json"]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_path)
        execution_time = time.time() - start_time
        
        metrics = {
            "line_coverage": 0.0,
            "branch_coverage": 0.0,
            "statement_coverage": 0.0,
            "function_coverage": 0.0,
            "execution_time": execution_time,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_total": 0
        }
        
        # Parse coverage summary from Jest output
        coverage_file = Path(project_path) / "coverage/coverage-summary.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                cov_data = json.load(f)
                if "total" in cov_data:
                    metrics["line_coverage"] = cov_data["total"]["lines"]["pct"]
                    metrics["branch_coverage"] = cov_data["total"]["branches"]["pct"]
                    metrics["statement_coverage"] = cov_data["total"]["statements"]["pct"]
                    metrics["function_coverage"] = cov_data["total"]["functions"]["pct"]
        
        return metrics
    
    def count_assertions(self, test_file: str) -> int:
        """Count assertions in JavaScript test file."""
        count = 0
        assertion_patterns = [
            r'\bexpect\(',
            r'\.toBe\(',
            r'\.toEqual\(',
            r'\.toThrow\(',
            r'assert\.',
            r'\.should\.',
        ]
        
        with open(test_file, 'r') as f:
            content = f.read()
            for pattern in assertion_patterns:
                count += len(re.findall(pattern, content))
        return count
    
    def check_compilation(self, test_path: str) -> tuple[int, int]:
        """Check JavaScript test files for syntax errors."""
        test_files = list(Path(test_path).rglob("*.test.js"))
        test_files.extend(Path(test_path).rglob("*.spec.js"))
        total = len(test_files)
        compilable = 0
        
        for test_file in test_files:
            cmd = ["node", "--check", str(test_file)]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                compilable += 1
        
        return total, compilable


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Main class to run benchmarks on AI testing bots."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzers = {
            "python": PythonAnalyzer(),
            "java": JavaAnalyzer(),
            "javascript": JavaScriptAnalyzer()
        }
    
    def run_creation_benchmark(
        self,
        bot_name: str,
        language: str,
        project_path: str,
        generated_test_path: str,
        generation_time: float = 0.0
    ) -> TestCreationMetrics:
        """
        Run test creation benchmark.
        
        Args:
            bot_name: Name of the AI bot being tested
            language: Programming language (python, java, javascript)
            project_path: Path to the source code project
            generated_test_path: Path where bot generated tests
            generation_time: Time taken to generate tests (measured externally)
        """
        analyzer = self.analyzers.get(language)
        if not analyzer:
            raise ValueError(f"Unsupported language: {language}")
        
        metrics = TestCreationMetrics()
        metrics.generation_time_seconds = generation_time
        
        # Check compilation
        total_tests, compilable = analyzer.check_compilation(generated_test_path)
        metrics.tests_generated = total_tests
        metrics.tests_compilable = compilable
        
        # Run coverage analysis
        cov_results = analyzer.run_coverage(project_path, generated_test_path)
        metrics.line_coverage = cov_results.get("line_coverage", 0.0)
        metrics.branch_coverage = cov_results.get("branch_coverage", 0.0)
        metrics.statement_coverage = cov_results.get("statement_coverage", 0.0)
        metrics.function_coverage = cov_results.get("function_coverage", 0.0)
        metrics.tests_passing = cov_results.get("tests_passed", 0)
        
        # Count assertions
        test_path = Path(generated_test_path)
        test_files = list(test_path.rglob("*test*.py" if language == "python" else "*Test*.java" if language == "java" else "*.test.js"))
        
        tests_with_assertions = 0
        for tf in test_files:
            if analyzer.count_assertions(str(tf)) > 0:
                tests_with_assertions += 1
        metrics.tests_with_assertions = tests_with_assertions
        
        metrics.calculate_derived_metrics()
        return metrics
    
    def run_execution_benchmark(
        self,
        language: str,
        project_path: str,
        test_path: str,
        known_bugs: list[dict] = None,
        run_mutation_testing: bool = False
    ) -> TestExecutionMetrics:
        """
        Run test execution benchmark.
        
        Args:
            language: Programming language
            project_path: Path to source code
            test_path: Path to test suite
            known_bugs: List of known bugs with info for detection testing
            run_mutation_testing: Whether to run mutation testing for mutation score
        """
        analyzer = self.analyzers.get(language)
        metrics = TestExecutionMetrics()
        
        # Count test files and cases
        test_files = list(Path(test_path).rglob("*test*"))
        metrics.total_test_files = len(test_files)
        
        # Calculate suite size
        total_size = sum(f.stat().st_size for f in Path(test_path).rglob("*") if f.is_file())
        metrics.test_suite_size_kb = total_size / 1024
        
        # Run tests multiple times to detect flaky tests
        results = []
        for _ in range(3):
            cov_results = analyzer.run_coverage(project_path, test_path)
            results.append(cov_results)
        
        metrics.execution_time_seconds = results[0]["execution_time"]
        metrics.total_test_cases = results[0]["tests_total"]
        
        if metrics.total_test_cases > 0:
            metrics.avg_test_time_ms = (metrics.execution_time_seconds * 1000) / metrics.total_test_cases
        
        # Detect flaky tests (tests with inconsistent results across runs)
        if len(results) > 1:
            pass_counts = [r["tests_passed"] for r in results]
            if max(pass_counts) != min(pass_counts):
                metrics.flaky_test_count = max(pass_counts) - min(pass_counts)
        
        # Test isolation (simplified - would need more sophisticated analysis)
        metrics.test_isolation_score = 0.9 if metrics.flaky_test_count == 0 else 0.7
        
        # Bug detection (if known bugs provided)
        if known_bugs:
            metrics.bugs_detected = self._check_bug_detection(
                language, project_path, test_path, known_bugs
            )
            total_known = len(known_bugs)
            metrics.calculate_derived_metrics(total_known)
        
        # Mutation testing (optional, time-intensive)
        if run_mutation_testing:
            metrics.mutation_score = self._run_mutation_testing(language, project_path, test_path)
        
        return metrics
    
    def run_maintenance_benchmark(
        self,
        bot_name: str,
        language: str,
        broken_test_path: str,
        repair_function: callable,
        expected_fixes: list[dict] = None
    ) -> TestMaintenanceMetrics:
        """
        Run test maintenance benchmark.
        
        Args:
            bot_name: Name of the AI bot
            language: Programming language
            broken_test_path: Path to test files with known issues
            repair_function: Callable that takes (broken_code, error_msg) and returns fixed code
            expected_fixes: List of expected fix patterns for validation
        """
        analyzer = self.analyzers.get(language)
        metrics = TestMaintenanceMetrics()
        
        # Find broken tests
        broken_tests = self._identify_broken_tests(language, broken_test_path)
        metrics.total_broken_tests = len(broken_tests)
        
        start_time = time.time()
        
        for broken_test in broken_tests:
            metrics.repair_attempts += 1
            
            try:
                # Call the bot's repair function
                fixed_code = repair_function(
                    broken_test["code"],
                    broken_test["error_message"]
                )
                
                # Validate the fix
                if self._validate_fix(language, fixed_code, broken_test):
                    metrics.successful_repairs += 1
                    metrics.errors_detected += 1
                else:
                    metrics.partial_repairs += 1
                    metrics.errors_detected += 1
                    
            except Exception as e:
                metrics.failed_repairs += 1
        
        metrics.repair_time_seconds = time.time() - start_time
        
        # Check for regressions (would require running full test suite)
        metrics.regression_introduced = 0  # Simplified
        
        metrics.calculate_derived_metrics()
        return metrics
    
    def _identify_broken_tests(self, language: str, test_path: str) -> list[dict]:
        """Identify broken tests by running them and capturing errors."""
        broken = []
        test_files = list(Path(test_path).rglob("*test*"))
        
        for tf in test_files:
            result = None
            if language == "python":
                result = subprocess.run(
                    ["python", "-m", "pytest", str(tf), "-v"],
                    capture_output=True, text=True
                )
            elif language == "javascript":
                result = subprocess.run(
                    ["npm", "test", "--", str(tf)],
                    capture_output=True, text=True
                )
            
            if result and result.returncode != 0:
                broken.append({
                    "file": str(tf),
                    "code": tf.read_text(),
                    "error_message": result.stderr + result.stdout
                })
        
        return broken
    
    def _validate_fix(self, language: str, fixed_code: str, original: dict) -> bool:
        """Validate that a fix actually works."""
        # Write fixed code to temp file and test it
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py' if language == 'python' else '.js', delete=False) as f:
            f.write(fixed_code)
            temp_path = f.name
        
        try:
            if language == "python":
                result = subprocess.run(
                    ["python", "-m", "pytest", temp_path, "-v"],
                    capture_output=True
                )
            else:
                result = subprocess.run(
                    ["node", "--check", temp_path],
                    capture_output=True
                )
            return result.returncode == 0
        finally:
            os.unlink(temp_path)
    
    def _check_bug_detection(self, language: str, project_path: str, test_path: str, known_bugs: list) -> int:
        """Check how many known bugs the test suite detects."""
        detected = 0
        for bug in known_bugs:
            # This would involve injecting the bug and running tests
            # Simplified implementation
            pass
        return detected
    
    def _run_mutation_testing(self, language: str, project_path: str, test_path: str) -> float:
        """Run mutation testing to calculate mutation score."""
        if language == "python":
            # Use mutmut for Python
            result = subprocess.run(
                ["mutmut", "run", "--paths-to-mutate", project_path],
                capture_output=True, text=True, cwd=project_path
            )
            # Parse results
        elif language == "java":
            # Use PIT for Java
            pass
        elif language == "javascript":
            # Use Stryker for JavaScript
            pass
        return 0.0  # Placeholder
    
    def save_results(self, result: BenchmarkResult, filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            filename = f"{result.bot_name}_{result.language}_{result.project_name}_{result.timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
    
    def generate_comparison_report(self, results: list[BenchmarkResult]) -> dict:
        """Generate a comparison report across multiple bots."""
        report = {
            "summary": {
                "total_bots_tested": len(results),
                "languages_tested": list(set(r.language for r in results)),
                "timestamp": datetime.now().isoformat()
            },
            "rankings": {
                "overall": [],
                "creation": [],
                "execution": [],
                "maintenance": [],
                "usability": []
            },
            "detailed_comparison": []
        }
        
        # Sort by different scores
        for category in ["overall", "creation", "execution", "maintenance", "usability"]:
            sorted_results = sorted(
                results,
                key=lambda r: getattr(r, f"{category}_score"),
                reverse=True
            )
            report["rankings"][category] = [
                {
                    "rank": i + 1,
                    "bot": r.bot_name,
                    "score": round(getattr(r, f"{category}_score"), 2)
                }
                for i, r in enumerate(sorted_results)
            ]
        
        return report


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_benchmark_config() -> dict:
    """Create a sample configuration for running benchmarks."""
    return {
        "bots": [
            {
                "name": "GitHub Copilot",
                "api_key_env": "COPILOT_API_KEY",
                "supported_languages": ["python", "java", "javascript"]
            },
            {
                "name": "Amazon CodeWhisperer",
                "api_key_env": "CODEWHISPERER_API_KEY",
                "supported_languages": ["python", "java", "javascript"]
            },
            {
                "name": "TestPilot",
                "api_key_env": "OPENAI_API_KEY",
                "supported_languages": ["javascript"]
            }
        ],
        "benchmark_projects": {
            "python": [
                {"name": "requests", "repo": "https://github.com/psf/requests"},
                {"name": "flask", "repo": "https://github.com/pallets/flask"}
            ],
            "java": [
                {"name": "junit5", "repo": "https://github.com/junit-team/junit5"}
            ],
            "javascript": [
                {"name": "lodash", "repo": "https://github.com/lodash/lodash"},
                {"name": "axios", "repo": "https://github.com/axios/axios"}
            ]
        },
        "metrics_weights": {
            "creation": 0.35,
            "execution": 0.30,
            "maintenance": 0.20,
            "usability": 0.15
        }
    }


if __name__ == "__main__":
    # Example usage
    print("AI Testing Bot Benchmark Framework")
    print("=" * 50)
    
    # Create sample config
    config = create_sample_benchmark_config()
    print("\nSample Configuration:")
    print(json.dumps(config, indent=2))
    
    # Initialize runner
    runner = BenchmarkRunner()
    
    print("\n\nTo run benchmarks, use the BenchmarkRunner class:")
    print("  runner = BenchmarkRunner()")
    print("  metrics = runner.run_creation_benchmark(...)")
