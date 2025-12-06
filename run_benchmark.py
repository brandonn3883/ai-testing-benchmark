#!/usr/bin/env python3
"""
AI Testing Bot Benchmark Runner

Benchmarks ChatGPT, Claude, and Gemini for test generation, execution, and maintenance.

Usage:
    python run_benchmark.py --bot chatgpt --language python --project ./src
    python run_benchmark.py --all --language python --project ./src
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_bots import create_bot, get_available_bots, BotInterface
from measure_creation import TestCreationMeasurer
from measure_execution import TestExecutionMeasurer
from measure_maintenance import TestMaintenanceMeasurer
from generate_report import CSVReportGenerator


class BenchmarkRunner:
    """Runs benchmarks on AI testing bots."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.creation_measurer = TestCreationMeasurer(str(self.output_dir / "work_creation"))
        self.execution_measurer = TestExecutionMeasurer(str(self.output_dir / "work_execution"))
        self.maintenance_measurer = TestMaintenanceMeasurer(str(self.output_dir / "work_maintenance"))
        self.report_generator = CSVReportGenerator(str(self.output_dir))
    
    def run_benchmark(
        self,
        bot: BotInterface,
        project_path: str,
        language: str,
        project_name: str = None,
        run_creation: bool = True,
        run_execution: bool = True,
        run_maintenance: bool = True,
        verbose: bool = True,
        debug: bool = False
    ) -> dict:
        """
        Run benchmark on a single bot.
        
        Args:
            bot: The bot to benchmark
            project_path: Path to source code
            language: python, java, or javascript
            project_name: Name for reporting
            run_creation: Run test creation benchmark
            run_execution: Run test execution benchmark
            run_maintenance: Run test maintenance benchmark
            verbose: Print progress
            debug: Enable debug output
        """
        if project_name is None:
            # Derive project name from path, but handle common patterns
            path = Path(project_path).resolve()
            name = path.name
            
            # If the directory is named "src", "lib", or "source", use parent directory name
            if name.lower() in ("src", "lib", "source", "main", "app"):
                name = path.parent.name
            
            # If still generic, try going up one more level
            if name.lower() in ("src", "lib", "source", "main", "app", "python", "java", "javascript"):
                name = path.parent.parent.name
            
            project_name = name
        
        result = {
            "bot_name": bot.name,
            "language": language,
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "creation": None,
            "execution": None,
            "maintenance": None
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING: {bot.name}")
            print(f"Project: {project_path}")
            print(f"Language: {language}")
            print(f"{'='*60}")
        
        generated_test_dir = self.output_dir / "work_creation" / "generated_tests"
        
        # Phase 1: Test Creation
        if run_creation:
            if verbose:
                print("\n[1/3] Test Creation...")
            try:
                # Create a wrapper that passes the correct language to generate_tests
                # Python passes (code, module_name), Java/JS pass just (code)
                def make_generator(lang):
                    return lambda code, module_name=None: bot.generate_tests(code, module_name=module_name, language=lang)
                
                if language == "python":
                    result["creation"] = self.creation_measurer.measure_python(
                        bot_name=bot.name,
                        project_path=project_path,
                        generate_tests=make_generator("python"),
                        project_name=project_name,
                        debug=debug
                    )
                elif language == "java":
                    result["creation"] = self.creation_measurer.measure_java(
                        bot_name=bot.name,
                        project_path=project_path,
                        generate_tests=make_generator("java"),
                        project_name=project_name,
                        debug=debug
                    )
                elif language == "javascript":
                    result["creation"] = self.creation_measurer.measure_javascript(
                        bot_name=bot.name,
                        project_path=project_path,
                        generate_tests=make_generator("javascript"),
                        project_name=project_name,
                        debug=debug
                    )
                if verbose and result["creation"]:
                    print(f"    Coverage: {result['creation'].line_coverage:.1f}%")
            except Exception as e:
                print(f"    Error: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
        
        # Phase 2: Test Maintenance (fix broken tests before mutation testing)
        if run_maintenance:
            if verbose:
                print("\n[2/3] Test Maintenance...")
            try:
                if generated_test_dir.exists():
                    if language == "python":
                        result["maintenance"] = self.maintenance_measurer.measure_python(
                            bot_name=bot.name,
                            repair_function=bot.repair_test,
                            test_dir=str(generated_test_dir),
                            source_dir=project_path,
                            project_name=project_name,
                            debug=debug
                        )
                    elif language == "java":
                        result["maintenance"] = self.maintenance_measurer.measure_java(
                            bot_name=bot.name,
                            repair_function=bot.repair_test,
                            test_dir=str(generated_test_dir),
                            source_dir=project_path,
                            project_name=project_name,
                            debug=debug
                        )
                    elif language == "javascript":
                        result["maintenance"] = self.maintenance_measurer.measure_javascript(
                            bot_name=bot.name,
                            repair_function=bot.repair_test,
                            test_dir=str(generated_test_dir),
                            source_dir=project_path,
                            project_name=project_name,
                            debug=debug
                        )
                    
                    if verbose and result["maintenance"]:
                        m = result["maintenance"]
                        if m.total_broken_tests == 0:
                            print(f"    No errors to fix (all tests pass)")
                        else:
                            print(f"    Fixed: {m.successful_repairs}/{m.total_broken_tests} ({m.get_fix_percentage():.1f}%)")
                            if m.successful_repairs > 0:
                                print(f"    First-attempt fixes: {m.first_attempt_fixes}/{m.successful_repairs}")
                                avg = m.get_avg_attempts_per_fix()
                                if avg:
                                    print(f"    Avg attempts per fix: {avg:.1f}")
                                
                                # Re-analyze coverage after fixes
                                if verbose:
                                    print("    Re-analyzing coverage after fixes...")
                                new_coverage = self._reanalyze_coverage(
                                    language, generated_test_dir, project_path, debug
                                )
                                if new_coverage is not None and result["creation"]:
                                    old_coverage = result["creation"].line_coverage
                                    result["creation"].line_coverage = new_coverage
                                    if verbose:
                                        print(f"    Coverage: {old_coverage:.1f}% -> {new_coverage:.1f}%")
                else:
                    if verbose:
                        print("    Skipped (no generated tests)")
            except Exception as e:
                print(f"    Error: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
        
        # Phase 3: Test Execution with Mutation Testing (on fixed tests)
        if run_execution:
            if verbose:
                print("\n[3/3] Test Execution (Mutation Testing)...")
            try:
                if generated_test_dir.exists():
                    if language == "python":
                        result["execution"] = self.execution_measurer.measure_python(
                            bot_name=bot.name,
                            project_path=project_path,
                            test_path=str(generated_test_dir),
                            project_name=project_name,
                            debug=debug
                        )
                    elif language == "java":
                        result["execution"] = self.execution_measurer.measure_java(
                            bot_name=bot.name,
                            project_path=project_path,
                            test_path=str(generated_test_dir),
                            project_name=project_name,
                            debug=debug
                        )
                    elif language == "javascript":
                        result["execution"] = self.execution_measurer.measure_javascript(
                            bot_name=bot.name,
                            project_path=project_path,
                            test_path=str(generated_test_dir),
                            project_name=project_name,
                            debug=debug
                        )
                    
                    if verbose and result["execution"]:
                        e = result["execution"]
                        print(f"    Tests: {e.tests_passed} passed, {e.tests_failed} failed")
                        if e.total_mutants > 0:
                            print(f"    Mutants: {e.mutants_killed}/{e.total_mutants} killed ({e.mutation_score:.1f}%)")
                        else:
                            print(f"    Mutants: Skipped (no passing tests)")
                else:
                    if verbose:
                        print("    Skipped (no generated tests)")
            except Exception as e:
                print(f"    Error: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
        
        # Display results
        if verbose:
            print(f"\n{'─'*40}")
            print("RESULTS:")
            if result["creation"]:
                print(f"  Test Coverage:      {result['creation'].line_coverage:.1f}%")
            if result["execution"]:
                e = result["execution"]
                print(f"  Mutation Score:     {e.mutation_score:.1f}% ({e.mutants_killed}/{e.total_mutants} killed)")
                print(f"  Surviving Mutants:  {e.mutants_survived} (false negatives)")
            if result["maintenance"]:
                m = result["maintenance"]
                print(f"  Errors Fixed:       {m.get_fix_percentage_display()}")
                if m.successful_repairs > 0:
                    eff = m.get_efficiency_score()
                    if eff is not None:
                        print(f"  Fix Efficiency:     {eff:.1f}% (first-attempt)")
        
        # Add to report generator
        self.report_generator.add_result(
            bot_name=result["bot_name"],
            language=result["language"],
            project_name=result["project_name"],
            creation=result["creation"],
            execution=result["execution"],
            maintenance=result["maintenance"]
        )
        
        return result
    
    def _reanalyze_coverage(
        self,
        language: str,
        test_dir: Path,
        source_dir: str,
        debug: bool = False
    ) -> float:
        """Re-analyze coverage after maintenance fixes."""
        import subprocess
        import re
        import platform
        import shutil
        
        try:
            if language == "python":
                # Run pytest with coverage
                result = subprocess.run(
                    ["python", "-m", "pytest", "--cov=.", "--cov-report=term-missing", "-q"],
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=120,
                    encoding='utf-8',
                    errors='replace'
                )
                output = (result.stdout or "") + (result.stderr or "")
                
                # Parse coverage: TOTAL ... XX%
                cov_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
                coverage_value = float(cov_match.group(1)) if cov_match else None
                
                # Cleanup coverage files
                coverage_file = test_dir / ".coverage"
                if coverage_file.exists():
                    coverage_file.unlink()
                htmlcov_dir = test_dir / "htmlcov"
                if htmlcov_dir.exists():
                    shutil.rmtree(htmlcov_dir, ignore_errors=True)
                
                return coverage_value
                    
            elif language == "java":
                # Run Maven test (JaCoCo will update coverage)
                is_windows = platform.system() == "Windows"
                mvn_cmd = "mvn.cmd" if is_windows else "mvn"
                
                result = subprocess.run(
                    [mvn_cmd, "test", "-q"],
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=300,
                    encoding='utf-8',
                    errors='replace'
                )
                
                coverage_value = None
                
                # Parse JaCoCo CSV report
                coverage_csv = test_dir / "target" / "site" / "jacoco" / "jacoco.csv"
                if coverage_csv.exists():
                    coverage_data = coverage_csv.read_text(encoding='utf-8')
                    lines = coverage_data.strip().split('\n')
                    if len(lines) > 1:
                        total_covered = 0
                        total_missed = 0
                        for line in lines[1:]:
                            parts = line.split(',')
                            if len(parts) >= 9:
                                try:
                                    missed = int(parts[7])
                                    covered = int(parts[8])
                                    total_missed += missed
                                    total_covered += covered
                                except (ValueError, IndexError):
                                    pass
                        
                        total = total_covered + total_missed
                        if total > 0:
                            coverage_value = (total_covered / total) * 100
                    
                    # Cleanup JaCoCo files
                    coverage_csv.unlink()
                
                # Also cleanup jacoco.exec
                jacoco_exec = test_dir / "target" / "jacoco.exec"
                if jacoco_exec.exists():
                    jacoco_exec.unlink()
                
                return coverage_value
                            
            elif language == "javascript":
                # Run Jest with coverage
                is_windows = platform.system() == "Windows"
                npx_cmd = "npx.cmd" if is_windows else "npx"
                
                result = subprocess.run(
                    [npx_cmd, "jest", "--coverage", "--coverageReporters=json-summary", "-silent"],
                    capture_output=True,
                    text=True,
                    cwd=str(test_dir),
                    timeout=120,
                    encoding='utf-8',
                    errors='replace'
                )
                
                coverage_value = None
                
                # Parse coverage-summary.json
                import json
                coverage_file = test_dir / "coverage" / "coverage-summary.json"
                if coverage_file.exists():
                    data = json.loads(coverage_file.read_text(encoding='utf-8'))
                    lines = data.get("total", {}).get("lines", {})
                    pct = lines.get("pct", 0)
                    if isinstance(pct, (int, float)):
                        coverage_value = float(pct)
                
                # Cleanup coverage directory
                coverage_dir = test_dir / "coverage"
                if coverage_dir.exists():
                    shutil.rmtree(coverage_dir, ignore_errors=True)
                
                return coverage_value
                        
        except Exception as e:
            if debug:
                print(f"    [DEBUG] Coverage re-analysis error: {e}")
        
        return None
    
    def run_all_bots(
        self,
        project_path: str,
        language: str,
        project_name: str = None,
        verbose: bool = True,
        debug: bool = False
    ) -> list:
        """Run benchmark on all available bots."""
        available = get_available_bots()
        
        if not available:
            print("No bots available. Set API keys:")
            print("  export OPENAI_API_KEY='...'")
            print("  export ANTHROPIC_API_KEY='...'")
            print("  export GOOGLE_API_KEY='...'")
            return []
        
        if verbose:
            print(f"\nAvailable bots: {[name for _, name in available]}")
        
        results = []
        for bot_id, bot_name in available:
            try:
                bot = create_bot(bot_id, debug=debug)
                result = self.run_benchmark(
                    bot=bot,
                    project_path=project_path,
                    language=language,
                    project_name=project_name,
                    verbose=verbose,
                    debug=debug
                )
                results.append(result)
            except Exception as e:
                print(f"Error with {bot_name}: {e}")
        
        return results
    
    def save_results(self) -> dict:
        """Save all results to CSV files."""
        return self.report_generator.generate_all_csvs()


def discover_projects(language: str, base_dir: Path = None) -> list[Path]:
    """
    Discover all projects for a given language.
    
    Projects are expected to be in:
    - ./python/<project_name>/src for Python
    - ./java/<project_name>/src for Java  
    - ./js/<project_name>/src for JavaScript
    
    Returns list of paths to project source directories.
    """
    if base_dir is None:
        base_dir = Path(".")
    
    # Map language to directory name
    lang_dirs = {
        "python": "python",
        "java": "java",
        "javascript": "js"
    }
    
    lang_dir = base_dir / lang_dirs.get(language, language)
    
    if not lang_dir.exists():
        return []
    
    projects = []
    
    # Look for project directories
    for item in lang_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check for src subdirectory first
            src_dir = item / "src"
            if src_dir.exists() and src_dir.is_dir():
                projects.append(src_dir)
            # Also check for lib subdirectory
            elif (item / "lib").exists():
                projects.append(item / "lib")
            # Otherwise use the project directory itself if it has source files
            else:
                # Check if directory has source files
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
    parser.add_argument("--bot", type=str, choices=["chatgpt", "claude", "gemini"],
                        help="Bot to benchmark")
    parser.add_argument("--all", action="store_true", help="Benchmark all available bots")
    parser.add_argument("--language", type=str, required=True,
                        choices=["python", "java", "javascript"], help="Programming language")
    parser.add_argument("--project", type=str, default=None,
                        help="Path to source code (if not specified, runs all projects in ./<language>/ folder)")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for API calls")
    
    args = parser.parse_args()
    
    if args.debug:
        print(f"[DEBUG] Args: bot={args.bot}, all={args.all}, language={args.language}, project={args.project}")
    
    if not args.bot and not args.all:
        parser.error("Specify --bot or --all")
    
    # Discover projects if not specified
    if args.project:
        project_paths = [Path(args.project)]
        if not project_paths[0].exists():
            parser.error(f"Project path does not exist: {args.project}")
    else:
        project_paths = discover_projects(args.language)
        if not project_paths:
            parser.error(f"No projects found in ./{args.language}/ directory. "
                        f"Create projects as ./{args.language}/<project_name>/src/ or use --project flag.")
        print(f"\nDiscovered {len(project_paths)} project(s):")
        for p in project_paths:
            print(f"  - {p}")
        print()
    
    # Create runner
    try:
        runner = BenchmarkRunner(output_dir=args.output)
    except Exception as e:
        print(f"Error creating BenchmarkRunner: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # Run benchmarks for each project
    for project_path in project_paths:
        if args.debug:
            print(f"\n[DEBUG] Processing project: {project_path}")
            print(f"[DEBUG] Files in project:")
            for f in project_path.iterdir():
                print(f"[DEBUG]   {f.name}")
        
        if args.all:
            try:
                runner.run_all_bots(
                    project_path=str(project_path),
                    language=args.language,
                    verbose=not args.quiet,
                    debug=args.debug
                )
            except Exception as e:
                print(f"Error running all bots on {project_path}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
        else:
            try:
                bot = create_bot(args.bot, debug=args.debug)
                runner.run_benchmark(
                    bot=bot,
                    project_path=str(project_path),
                    language=args.language,
                    verbose=not args.quiet,
                    debug=args.debug
                )
            except Exception as e:
                print(f"Error running {args.bot} on {project_path}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
    
    # Save CSV results
    if args.debug:
        print("[DEBUG] Saving results...")
    try:
        csv_files = runner.save_results()
        print(f"\n{'='*60}")
        print("CSV FILES GENERATED:")
        for name, path in csv_files.items():
            print(f"  {name}: {path}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"[DEBUG] Error saving results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()