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
                
            original_code = source_file.read_text(encoding='utf-8', errors='replace')
            
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
        """Generate mutants from source code with randomization for variety."""
        import random
        
        all_mutants = []
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
                            all_mutants.append((mutated_code, desc, line_num))
                        except SyntaxError:
                            pass  # Invalid mutation, skip
        
        # Shuffle to get variety, then take max_count
        random.shuffle(all_mutants)
        return all_mutants[:max_count]
    
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
            test_content = test_file.read_text(encoding='utf-8', errors='replace')
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
            test_content = test_file.read_text(encoding='utf-8', errors='replace')
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
    
    # JavaScript mutation operators (still needed for manual JS mutation testing)
    JS_MUTATION_OPERATORS = [
        # Arithmetic operators
        (r'(\s)\+(\s)', r'\1-\2', 'arithmetic: + to -'),
        (r'(\s)-(\s)', r'\1+\2', 'arithmetic: - to +'),
        (r'(\s)\*(\s)', r'\1/\2', 'arithmetic: * to /'),
        (r'(\s)/(\s)', r'\1*\2', 'arithmetic: / to *'),
        
        # Comparison operators
        (r'===', r'!==', 'comparison: === to !=='),
        (r'!==', r'===', 'comparison: !== to ==='),
        (r'==', r'!=', 'comparison: == to !='),
        (r'!=', r'==', 'comparison: != to =='),
        (r'<=', r'>', 'comparison: <= to >'),
        (r'>=', r'<', 'comparison: >= to <'),
        
        # Logical operators
        (r'&&', r'||', 'logical: && to ||'),
        (r'\|\|', r'&&', 'logical: || to &&'),
        (r'!(?!=)', r'', 'logical: remove !'),
        
        # Boolean literals
        (r'\btrue\b', r'false', 'boolean: true to false'),
        (r'\bfalse\b', r'true', 'boolean: false to true'),
        
        # Boundary mutations
        (r'\b0\b', r'1', 'boundary: 0 to 1'),
        (r'\b1\b', r'0', 'boundary: 1 to 0'),
        
        # Return value mutations
        (r'return\s+(.+);', r'return null;', 'return: value to null'),
    ]
    
    def measure_java(
        self,
        bot_name: str,
        project_path: str,
        test_path: str,
        project_name: str = "unnamed",
        max_mutants: int = 20,
        debug: bool = False
    ) -> ExecutionMeasurement:
        """Measure Java test execution using PITest mutation testing."""
        measurement = ExecutionMeasurement(
            bot_name=bot_name,
            language="java",
            project_name=project_name
        )
        
        project_path = Path(project_path).resolve()
        test_path = Path(test_path).resolve()
        
        import platform
        is_windows = platform.system() == "Windows"
        mvn_cmd = "mvn.cmd" if is_windows else "mvn"
        
        # Check for Maven project structure
        pom_file = test_path / "pom.xml"
        if not pom_file.exists():
            if debug:
                print("    [DEBUG] No pom.xml found, skipping Java mutation testing")
            return measurement
        
        # Add PITest plugin to pom.xml if not present
        self._ensure_pitest_plugin(pom_file, debug)
        
        # Run baseline tests with Maven
        if debug:
            print("    [DEBUG] Running baseline Java tests...")
        
        baseline = self._run_java_tests_maven(test_path, mvn_cmd, is_windows, debug)
        measurement.total_test_cases = baseline["total"]
        measurement.tests_passed = baseline["passed"]
        measurement.tests_failed = baseline["failed"]
        
        if baseline["passed"] == 0:
            if debug:
                print("    [DEBUG] No passing tests, skipping mutation testing")
            return measurement
        
        # Run PITest mutation testing
        if debug:
            print("    [DEBUG] Running PITest mutation testing...")
        
        mutation_results = self._run_pitest(test_path, mvn_cmd, is_windows, debug)
        
        measurement.total_mutants = mutation_results["total"]
        measurement.mutants_killed = mutation_results["killed"]
        measurement.mutants_survived = mutation_results["survived"]
        measurement.calculate_mutation_score()
        
        if debug:
            print(f"    [DEBUG] Mutation score: {measurement.mutation_score:.1f}%")
        
        return measurement
    
    def _ensure_pitest_plugin(self, pom_file: Path, debug: bool = False):
        """Add PITest plugin to pom.xml if not already present."""
        content = pom_file.read_text(encoding='utf-8', errors='replace')
        
        if 'pitest-maven' in content:
            if debug:
                print("    [DEBUG] PITest plugin already in pom.xml")
            return
        
        if debug:
            print("    [DEBUG] Adding PITest plugin to pom.xml")
        
        # Find source class names from src/main/java
        main_java = pom_file.parent / "src" / "main" / "java"
        source_classes = []
        if main_java.exists():
            for java_file in main_java.glob("*.java"):
                # Get class name from file
                class_name = java_file.stem
                source_classes.append(class_name)
        
        if not source_classes:
            source_classes = ["*"]  # Fallback, but this might cause issues
        
        # Build targetClasses XML
        target_classes_xml = '\n'.join(f'                        <param>{cls}</param>' for cls in source_classes)
        
        pitest_plugin = f'''
            <!-- PITest mutation testing plugin -->
            <plugin>
                <groupId>org.pitest</groupId>
                <artifactId>pitest-maven</artifactId>
                <version>1.15.0</version>
                <dependencies>
                    <dependency>
                        <groupId>org.pitest</groupId>
                        <artifactId>pitest-junit5-plugin</artifactId>
                        <version>1.2.1</version>
                    </dependency>
                </dependencies>
                <configuration>
                    <targetClasses>
{target_classes_xml}
                    </targetClasses>
                    <targetTests>
                        <param>*Test</param>
                    </targetTests>
                    <outputFormats>
                        <outputFormat>CSV</outputFormat>
                        <outputFormat>HTML</outputFormat>
                    </outputFormats>
                    <timestampedReports>false</timestampedReports>
                    <failWhenNoMutations>false</failWhenNoMutations>
                </configuration>
            </plugin>
'''
        
        if debug:
            print(f"    [DEBUG] Target classes: {source_classes}")
        
        # Insert before </plugins>
        if '</plugins>' in content:
            content = content.replace('</plugins>', pitest_plugin + '        </plugins>')
            pom_file.write_text(content, encoding='utf-8')
        elif '</build>' in content:
            # No plugins section, add one
            plugins_section = f'''    <plugins>
{pitest_plugin}
        </plugins>
    '''
            content = content.replace('</build>', plugins_section + '</build>')
            pom_file.write_text(content, encoding='utf-8')
    
    def _run_pitest(self, test_dir: Path, mvn_cmd: str, is_windows: bool, debug: bool = False) -> dict:
        """Run PITest mutation testing and return results."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        try:
            # Run PITest
            result = subprocess.run(
                [mvn_cmd, "org.pitest:pitest-maven:mutationCoverage", "-q"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=600,  # 10 minute timeout for mutation testing
                encoding='utf-8',
                errors='replace'
            )
            
            output = (result.stdout or "") + (result.stderr or "")
            
            if debug:
                print(f"    [DEBUG] PITest output (last 1000 chars): {output[-1000:]}")
            
            # Parse results from CSV report
            # PITest creates reports in target/pit-reports/
            pit_reports = test_dir / "target" / "pit-reports"
            
            # Find the mutations.csv file
            csv_file = pit_reports / "mutations.csv"
            if csv_file.exists():
                results = self._parse_pitest_csv(csv_file, debug)
            else:
                # Try to find in subdirectory (timestamped reports)
                for subdir in pit_reports.glob("*"):
                    if subdir.is_dir():
                        csv_file = subdir / "mutations.csv"
                        if csv_file.exists():
                            results = self._parse_pitest_csv(csv_file, debug)
                            break
            
            # If CSV not found, try parsing from console output
            if results["total"] == 0:
                results = self._parse_pitest_output(output, debug)
                
        except subprocess.TimeoutExpired:
            if debug:
                print("    [DEBUG] PITest timed out")
        except Exception as e:
            if debug:
                print(f"    [DEBUG] PITest error: {e}")
        
        return results
    
    def _parse_pitest_csv(self, csv_file: Path, debug: bool = False) -> dict:
        """Parse PITest mutations.csv report."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        try:
            content = csv_file.read_text(encoding='utf-8', errors='replace')
            lines = content.strip().split('\n')
            
            for line in lines[1:]:  # Skip header
                results["total"] += 1
                # CSV format: file,class,mutator,method,line,status,killingTest
                parts = line.split(',')
                if len(parts) >= 6:
                    status = parts[5].strip().upper()
                    if status in ['KILLED', 'TIMED_OUT']:
                        results["killed"] += 1
                    elif status in ['SURVIVED', 'NO_COVERAGE']:
                        results["survived"] += 1
            
            if debug:
                print(f"    [DEBUG] PITest CSV results: {results}")
                
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Error parsing PITest CSV: {e}")
        
        return results
    
    def _parse_pitest_output(self, output: str, debug: bool = False) -> dict:
        """Parse PITest results from console output."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        # Look for summary line like:
        # >> Generated 45 mutations Killed 38 (84%)
        # or: mutations: 45, killed: 38
        
        generated_match = re.search(r'Generated\s+(\d+)\s+mutations', output)
        killed_match = re.search(r'Killed\s+(\d+)', output)
        
        if generated_match:
            results["total"] = int(generated_match.group(1))
        if killed_match:
            results["killed"] = int(killed_match.group(1))
        
        results["survived"] = results["total"] - results["killed"]
        
        # Alternative pattern
        if results["total"] == 0:
            alt_match = re.search(r'(\d+)\s+mutations?\s+(?:were\s+)?(?:generated|created)', output, re.IGNORECASE)
            if alt_match:
                results["total"] = int(alt_match.group(1))
            
            killed_alt = re.search(r'(\d+)\s+(?:mutations?\s+)?killed', output, re.IGNORECASE)
            if killed_alt:
                results["killed"] = int(killed_alt.group(1))
                results["survived"] = results["total"] - results["killed"]
        
        if debug:
            print(f"    [DEBUG] PITest parsed from output: {results}")
        
        return results
    
    def _run_java_tests_maven(self, test_dir: Path, mvn_cmd: str, is_windows: bool, debug: bool = False) -> dict:
        """Run Maven tests and return results."""
        results = {"total": 0, "passed": 0, "failed": 0, "duration": 0.0}
        
        try:
            result = subprocess.run(
                [mvn_cmd, "test", "-q"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            output = (result.stdout or "") + (result.stderr or "")
            
            # Parse: Tests run: 5, Failures: 1, Errors: 0, Skipped: 0
            tests_match = re.search(r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", output)
            if tests_match:
                total = int(tests_match.group(1))
                failures = int(tests_match.group(2))
                errors = int(tests_match.group(3))
                results["total"] = total
                results["passed"] = total - failures - errors
                results["failed"] = failures + errors
            
            if debug:
                print(f"    [DEBUG] Maven test results: {results}")
                
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Maven test error: {e}")
        
        return results
    
    def measure_javascript(
        self,
        bot_name: str,
        project_path: str,
        test_path: str,
        project_name: str = "unnamed",
        max_mutants: int = 20,
        debug: bool = False
    ) -> ExecutionMeasurement:
        """Measure JavaScript test execution using mutation testing."""
        measurement = ExecutionMeasurement(
            bot_name=bot_name,
            language="javascript",
            project_name=project_name
        )
        
        project_path = Path(project_path).resolve()
        test_path = Path(test_path).resolve()
        
        # Run baseline tests
        if debug:
            print("    [DEBUG] Running baseline JavaScript tests...")
        
        baseline = self._run_javascript_tests(test_path, debug)
        measurement.total_test_cases = baseline["total"]
        measurement.tests_passed = baseline["passed"]
        measurement.tests_failed = baseline["failed"]
        
        if baseline["passed"] == 0:
            if debug:
                print("    [DEBUG] No passing tests, skipping mutation testing")
            return measurement
        
        # Run manual mutation testing
        if debug:
            print("    [DEBUG] Starting JavaScript mutation testing...")
        
        # Only mutate source files that exist in the original project_path
        # This prevents mutating leftover files from previous runs
        original_source_files = [f.name for f in project_path.glob("*.js") 
                                 if ".test." not in f.name and ".spec." not in f.name]
        
        source_files = [f for f in test_path.glob("*.js") 
                       if ".test." not in f.name and ".spec." not in f.name 
                       and "jest.config" not in f.name
                       and f.name in original_source_files]  # Only files from current project
        
        if debug:
            print(f"    [DEBUG] Found {len(source_files)} source file(s) for mutation")
            for sf in source_files:
                print(f"    [DEBUG]   - {sf.name}")
        
        if not source_files:
            if debug:
                print("    [DEBUG] No source files found for mutation testing")
            return measurement
        
        mutation_results = self._run_manual_js_mutation_testing(
            source_files, test_path, max_mutants, debug
        )
        
        if debug:
            print(f"    [DEBUG] Mutation results: {mutation_results}")
        
        measurement.total_mutants = mutation_results["total"]
        measurement.mutants_killed = mutation_results["killed"]
        measurement.mutants_survived = mutation_results["survived"]
        measurement.calculate_mutation_score()
        
        if debug:
            print(f"    [DEBUG] Mutation score: {measurement.mutation_score:.1f}%")
        
        return measurement
    
    def _run_javascript_tests(self, test_dir: Path, debug: bool = False) -> dict:
        """Run Jest tests and return results."""
        results = {"total": 0, "passed": 0, "failed": 0, "duration": 0.0}
        
        # Determine correct command for Windows vs Unix
        import platform
        is_windows = platform.system() == "Windows"
        npx_cmd = "npx.cmd" if is_windows else "npx"
        
        try:
            result = subprocess.run(
                [npx_cmd, "jest", "--no-colors", "--passWithNoTests"],
                capture_output=True,
                text=True,
                cwd=str(test_dir),
                timeout=120,
                shell=is_windows,
                encoding='utf-8',
                errors='replace'
            )
            
            output = (result.stdout or "") + (result.stderr or "")
            
            # Parse Jest output
            passed = re.search(r"(\d+) passed", output)
            failed = re.search(r"(\d+) failed", output)
            
            results["passed"] = int(passed.group(1)) if passed else 0
            results["failed"] = int(failed.group(1)) if failed else 0
            results["total"] = results["passed"] + results["failed"]
            
            if debug:
                print(f"    [DEBUG] Jest results: {results}")
            
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Jest error: {e}")
        
        return results
    
    def _run_manual_js_mutation_testing(
        self,
        source_files: list,
        test_dir: Path,
        max_mutants: int,
        debug: bool
    ) -> dict:
        """Run manual mutation testing on JavaScript source files."""
        results = {"total": 0, "killed": 0, "survived": 0}
        
        mutation_dir = self.work_dir / "mutations"
        mutation_dir.mkdir(parents=True, exist_ok=True)
        
        import platform
        is_windows = platform.system() == "Windows"
        npx_cmd = "npx.cmd" if is_windows else "npx"
        
        for source_file in source_files:
            code = source_file.read_text(encoding='utf-8', errors='replace')
            mutants = self._generate_javascript_mutants(code, max_mutants - results["total"])
            
            for mutant_code, desc, line_num in mutants:
                if results["total"] >= max_mutants:
                    break
                
                results["total"] += 1
                
                if debug:
                    print(f"    [DEBUG] Testing mutant {results['total']}: {desc} at line {line_num}")
                
                killed = self._test_javascript_mutant(
                    source_file, mutant_code, test_dir, mutation_dir, npx_cmd, is_windows, debug
                )
                
                if killed:
                    results["killed"] += 1
                else:
                    results["survived"] += 1
        
        return results
    
    def _generate_javascript_mutants(self, code: str, max_count: int) -> list:
        """Generate JavaScript mutants with randomization for variety."""
        import random
        
        all_mutants = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            if stripped.startswith('import ') or ('require' in stripped and 'const' in stripped):
                continue
            
            for pattern, replacement, desc in self.JS_MUTATION_OPERATORS:
                if re.search(pattern, line):
                    mutated_line = re.sub(pattern, replacement, line, count=1)
                    if mutated_line != line:
                        mutated_lines = lines.copy()
                        mutated_lines[line_num - 1] = mutated_line
                        all_mutants.append(('\n'.join(mutated_lines), desc, line_num))
        
        # Shuffle to get variety, then take max_count
        random.shuffle(all_mutants)
        return all_mutants[:max_count]
    
    def _test_javascript_mutant(
        self,
        source_file: Path,
        mutant_code: str,
        test_dir: Path,
        mutation_dir: Path,
        npx_cmd: str,
        is_windows: bool,
        debug: bool
    ) -> bool:
        """Test a JavaScript mutant. Returns True if killed."""
        # Clear mutation dir (except node_modules)
        for f in mutation_dir.glob("*"):
            if f.is_file():
                f.unlink()
            elif f.is_dir() and f.name not in ["node_modules", "coverage"]:
                shutil.rmtree(f)
        
        # Copy files
        for f in test_dir.glob("*.js"):
            shutil.copy(f, mutation_dir / f.name)
        
        # Copy config files
        for config in ["package.json", "jest.config.js"]:
            src = test_dir / config
            if src.exists():
                shutil.copy(src, mutation_dir / config)
        
        # Copy node_modules if not exists
        node_modules_src = test_dir / "node_modules"
        node_modules_dst = mutation_dir / "node_modules"
        if node_modules_src.exists() and not node_modules_dst.exists():
            shutil.copytree(node_modules_src, node_modules_dst)
        
        # Write mutant
        mutant_file = mutation_dir / source_file.name
        mutant_file.write_text(mutant_code, encoding='utf-8')
        
        # Run tests
        try:
            result = subprocess.run(
                [npx_cmd, "jest", "--no-colors", "--passWithNoTests"],
                capture_output=True,
                text=True,
                cwd=str(mutation_dir),
                timeout=30,
                shell=is_windows,
                encoding='utf-8',
                errors='replace'
            )
            return result.returncode != 0  # Tests failed = killed
        except subprocess.TimeoutExpired:
            return True
        except Exception:
            return True


if __name__ == "__main__":
    print("Test Execution Measurer")
    print("Measures test quality using mutation testing")
    print("Usage: measurer.measure_python(bot_name, project_path, test_path)")