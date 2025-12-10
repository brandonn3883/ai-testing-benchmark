#!/usr/bin/env python3
"""
CSV Report Generator for AI Testing Bot Benchmark

Generates simple CSV files for benchmark results.
"""

import csv
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from measure_creation import CreationMeasurement
from measure_execution import ExecutionMeasurement
from measure_maintenance import MaintenanceMeasurement


class CSVReportGenerator:
    """Generates CSV reports from benchmark results."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_result(self, bot_name: str, language: str, project_name: str,
                   creation: CreationMeasurement = None,
                   execution: ExecutionMeasurement = None,
                   maintenance: MaintenanceMeasurement = None):
        """Add a benchmark result."""
        self.results.append({
            "bot_name": bot_name,
            "language": language,
            "project_name": project_name,
            "timestamp": datetime.now().isoformat(),
            "creation": creation,
            "execution": execution,
            "maintenance": maintenance
        })
    
    def generate_summary_csv(self, filename: str = "benchmark_summary.csv") -> str:
        """Generate summary CSV with key metrics for all bots, one row per source file."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header - added file_name and file_coverage_pct
            writer.writerow([
                "bot_name", "language", "project", "file_name", "timestamp",
                "file_coverage_pct", "project_coverage_pct", "tests_failed",
                "mutation_score_pct", "mutants_killed", "mutants_survived", 
                "errors_fixed_pct", "failed_repairs", "errors_remain"
            ])
            
            # Data rows - one row per source file
            for r in self.results:
                project_coverage = r["creation"].line_coverage if r["creation"] else 0
                tests_failed = r["creation"].tests_failed if r["creation"] else 0
                per_file_coverage = r["creation"].per_file_coverage if r["creation"] else {}
                
                if r["execution"]:
                    mutation_score = r["execution"].mutation_score
                    killed = r["execution"].mutants_killed
                    survived = r["execution"].mutants_survived
                else:
                    mutation_score = 0
                    killed = 0
                    survived = 0
                
                # Handle N/A case for maintenance
                if r["maintenance"]:
                    fixed_pct = r["maintenance"].get_fix_percentage()
                    fixed_pct_str = "N/A" if fixed_pct is None else str(round(fixed_pct, 2))
                    failed_repairs = r["maintenance"].failed_repairs
                    errors_remain = r["maintenance"].errors_remain
                else:
                    fixed_pct_str = "N/A"
                    failed_repairs = 0
                    errors_remain = False
                
                # Write one row per source file
                if per_file_coverage:
                    for file_name, file_cov in per_file_coverage.items():
                        writer.writerow([
                            r["bot_name"], r["language"], r["project_name"], file_name, r["timestamp"],
                            round(file_cov, 2), round(project_coverage, 2), tests_failed,
                            round(mutation_score, 2), killed, survived, 
                            fixed_pct_str, failed_repairs, errors_remain
                        ])
                else:
                    # No per-file data, write single row with empty file_name
                    writer.writerow([
                        r["bot_name"], r["language"], r["project_name"], "", r["timestamp"],
                        round(project_coverage, 2), round(project_coverage, 2), tests_failed,
                        round(mutation_score, 2), killed, survived, 
                        fixed_pct_str, failed_repairs, errors_remain
                    ])
        
        return str(filepath)
    
    def generate_creation_csv(self, filename: str = "creation_metrics.csv") -> str:
        """Generate CSV for test creation metrics."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "bot_name", "language", "project",
                "source_files", "tests_generated", "tests_compilable", 
                "tests_passing", "tests_failed",
                "line_coverage_pct", "branch_coverage_pct", "generation_time_sec"
            ])
            
            for r in self.results:
                c = r["creation"]
                if c:
                    writer.writerow([
                        r["bot_name"], r["language"], r["project_name"],
                        c.source_files_count, c.tests_generated, c.tests_compilable, 
                        c.tests_passing, c.tests_failed,
                        round(c.line_coverage, 2), round(c.branch_coverage, 2),
                        round(c.generation_time_seconds, 2)
                    ])
        
        return str(filepath)
    
    def generate_execution_csv(self, filename: str = "execution_metrics.csv") -> str:
        """Generate CSV for test execution metrics (mutation testing)."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "bot_name", "language", "project",
                "total_tests", "tests_passed", "tests_failed",
                "total_mutants", "mutants_killed", "mutants_survived",
                "mutation_score_pct", "execution_time_sec"
            ])
            
            for r in self.results:
                e = r["execution"]
                if e:
                    writer.writerow([
                        r["bot_name"], r["language"], r["project_name"],
                        e.total_test_cases, e.tests_passed, e.tests_failed,
                        e.total_mutants, e.mutants_killed, e.mutants_survived,
                        round(e.mutation_score, 2),
                        round(e.total_execution_time_seconds, 2)
                    ])
        
        return str(filepath)
    
    def generate_maintenance_csv(self, filename: str = "maintenance_metrics.csv") -> str:
        """Generate CSV for test maintenance metrics."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "bot_name", "language", "project",
                "total_broken_tests", "successful_repairs", "failed_repairs",
                "errors_remain", "fix_percentage", "total_attempts", "first_attempt_fixes",
                "efficiency_pct", "avg_attempts_per_fix", "total_repair_time_sec"
            ])
            
            for r in self.results:
                m = r["maintenance"]
                if m:
                    fix_pct = m.get_fix_percentage()
                    fix_pct_str = "N/A" if fix_pct is None else str(round(fix_pct, 2))
                    
                    efficiency = m.get_efficiency_score()
                    efficiency_str = "N/A" if efficiency is None else str(round(efficiency, 2))
                    
                    avg_attempts = m.get_avg_attempts_per_fix()
                    avg_attempts_str = "N/A" if avg_attempts is None else str(round(avg_attempts, 2))
                    
                    writer.writerow([
                        r["bot_name"], r["language"], r["project_name"],
                        m.total_broken_tests, m.successful_repairs, m.failed_repairs,
                        m.errors_remain, fix_pct_str, m.total_repair_attempts, m.first_attempt_fixes,
                        efficiency_str, avg_attempts_str,
                        round(m.total_repair_time_seconds, 2)
                    ])
        
        return str(filepath)
    
    def generate_per_file_coverage_csv(self, filename: str = "per_file_coverage.csv") -> str:
        """Generate CSV for per-file coverage metrics."""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            writer.writerow([
                "bot_name", "language", "project", "source_file", "line_coverage_pct"
            ])
            
            for r in self.results:
                c = r["creation"]
                if c and c.per_file_coverage:
                    for source_file, coverage in c.per_file_coverage.items():
                        writer.writerow([
                            r["bot_name"], r["language"], r["project_name"],
                            source_file, coverage
                        ])
        
        return str(filepath)
    
    def generate_all_csvs(self) -> dict:
        """Generate all CSV reports."""
        return {
            "summary": self.generate_summary_csv(),
            "creation": self.generate_creation_csv(),
            "execution": self.generate_execution_csv(),
            "maintenance": self.generate_maintenance_csv(),
            "per_file_coverage": self.generate_per_file_coverage_csv()
        }
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        for r in self.results:
            coverage = r["creation"].line_coverage if r["creation"] else 0
            
            if r["execution"]:
                mutation_score = r["execution"].mutation_score
                survived = r["execution"].mutants_survived
            else:
                mutation_score = 0
                survived = 0
            
            if r["maintenance"]:
                fixed_display = r["maintenance"].get_fix_percentage_display()
            else:
                fixed_display = "N/A"
            
            print(f"\n{r['bot_name']} ({r['language']}):")
            print(f"  Test Coverage:      {coverage:.1f}%")
            print(f"  Mutation Score:     {mutation_score:.1f}%")
            print(f"  Surviving Mutants:  {survived} (false negatives)")
            print(f"  Errors Fixed:       {fixed_display}")


if __name__ == "__main__":
    print("CSV Report Generator")
    print("Use: generator = CSVReportGenerator()")
    print("     generator.add_result(...)")
    print("     generator.generate_all_csvs()")