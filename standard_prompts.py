#!/usr/bin/env python3
"""
Standardized Prompts for AI Testing Bot Benchmark

This module defines the EXACT prompts used across all LLMs to ensure
fair and consistent comparison. All bots receive identical instructions.

DO NOT modify these prompts without updating the documentation,
as changes will affect benchmark comparability.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StandardPrompts:
    """
    Standardized prompts for all three benchmark categories.
    These prompts are used identically across ChatGPT, Claude, Copilot, and Gemini.
    """
    
    # =========================================================================
    # TEST CREATION PROMPTS
    # =========================================================================
    
    @staticmethod
    def test_creation_system_prompt(language: str) -> str:
        """System prompt for test generation (used by all LLMs)."""
        
        framework_map = {
            "python": "pytest",
            "java": "JUnit 5",
            "javascript": "Jest with CommonJS"
        }
        framework = framework_map.get(language, "appropriate testing framework")
        
        # Language-specific instructions
        if language == "javascript":
            extra_instructions = """
7. Use CommonJS syntax (require/module.exports), NOT ES Modules (import/export)
8. Do NOT use: import { jest } from '@jest/globals' or any import statements
9. Use: const { functionName } = require('./moduleName'); for imports
10. Use: describe(), it(), expect() - these are available globally in Jest"""
        else:
            extra_instructions = ""
        
        return f"""You are a software testing expert. Your task is to generate comprehensive unit tests.

REQUIREMENTS:
1. Generate tests using {framework}
2. Achieve high code coverage (target >80% line coverage)
3. Include tests for:
   - Normal/expected behavior (happy path)
   - Edge cases (empty inputs, boundary values, zero, negative numbers)
   - Error handling (invalid inputs, exceptions)
   - Type validation where applicable
4. Use descriptive test names that explain what is being tested
5. Include meaningful assertions with clear expected values
6. Each test should be independent and not rely on other tests{extra_instructions}

OUTPUT FORMAT:
- Return ONLY valid, runnable test code
- Do NOT include any explanations or markdown formatting
- The output should be directly saveable as a test file and executable"""

    @staticmethod
    def test_creation_user_prompt(source_code: str, language: str, module_name: str = None) -> str:
        """User prompt for test generation (used by all LLMs)."""
        
        if language == "python" and module_name:
            import_instruction = f"\nIMPORTANT: Import the functions to test using: from {module_name} import *\n"
        elif language == "java" and module_name:
            import_instruction = f"\nIMPORTANT: Do not include a package declaration and do not import any other libraries other than those from the source code and JUnit\n"
        elif language == "javascript":
            if module_name:
                import_instruction = f"""
CRITICAL: Use CommonJS require syntax, NOT ES Module import syntax.
Example: const {{ functionName }} = require('./{module_name}');
Do NOT use: import {{ functionName }} from './{module_name}';
Do NOT use: import {{ jest }} from '@jest/globals';
"""
            else:
                import_instruction = """
CRITICAL: Use CommonJS require syntax, NOT ES Module import syntax.
Do NOT use any import statements. Use require() instead.
"""
        else:
            import_instruction = ""
        
        return f"""Generate comprehensive unit tests for the following {language} code:
{import_instruction}
{source_code}

Generate the complete test file now."""

    # =========================================================================
    # TEST EXECUTION / TEST SUITE PROMPTS
    # =========================================================================
    
    @staticmethod
    def test_suite_system_prompt(language: str) -> str:
        """System prompt for test suite organization."""
        
        return f"""You are a software testing expert specializing in test suite organization.

TASK: Create an optimized test suite configuration for {language} tests.

REQUIREMENTS:
1. Organize tests logically by module/functionality
2. Configure appropriate test runners and plugins
3. Set up coverage reporting
4. Enable parallel test execution where possible
5. Configure appropriate timeouts

OUTPUT FORMAT:
- Return ONLY the configuration file content
- Do NOT include explanations or markdown formatting"""

    @staticmethod
    def test_suite_user_prompt(test_files: list, language: str) -> str:
        """User prompt for test suite creation."""
        
        files_list = "\n".join(f"- {f}" for f in test_files)
        
        return f"""Create a test suite configuration for these {language} test files:

{files_list}

Generate the configuration now."""

    # =========================================================================
    # TEST MAINTENANCE / REPAIR PROMPTS
    # =========================================================================
    
    @staticmethod
    def test_repair_system_prompt() -> str:
        """System prompt for test repair (used by all LLMs)."""
        
        return """You are a software testing expert specializing in debugging and fixing broken tests.

TASK: Fix the broken test code based on the error message provided.

REQUIREMENTS:
1. Identify the root cause of the error
2. Fix ONLY the specific issue causing the failure
3. Maintain the original test intent and purpose
4. Do NOT add new tests or change unrelated code
5. Do NOT change the test logic unless it is the source of the error
6. Preserve test naming and structure

OUTPUT FORMAT:
- Return ONLY the corrected test code
- Do NOT include any explanations or markdown formatting
- Do NOT wrap the code in code blocks
- The output should be directly executable"""

    @staticmethod
    def test_repair_user_prompt(broken_code: str, error_message: str, error_type: str) -> str:
        """User prompt for test repair (used by all LLMs)."""
        
        return f"""Fix this broken test code.

ERROR TYPE: {error_type}

ERROR MESSAGE:
{error_message}

BROKEN CODE:
{broken_code}

Provide the fixed code now."""

    # =========================================================================
    # LANGUAGE-SPECIFIC EXAMPLES (included in prompts when helpful)
    # =========================================================================
    
    @staticmethod
    def get_test_example(language: str) -> str:
        """Get a minimal test example for the specified language."""
        
        examples = {
            "python": '''import pytest

def test_function_returns_expected_value():
    result = function_under_test(input_value)
    assert result == expected_value

def test_function_handles_empty_input():
    result = function_under_test([])
    assert result is None

def test_function_raises_on_invalid_input():
    with pytest.raises(ValueError):
        function_under_test(invalid_input)''',

            "java": '''import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FunctionTest {
    @Test
    void testFunctionReturnsExpectedValue() {
        int result = functionUnderTest(inputValue);
        assertEquals(expectedValue, result);
    }
    
    @Test
    void testFunctionHandlesEmptyInput() {
        assertNull(functionUnderTest(Collections.emptyList()));
    }
    
    @Test
    void testFunctionThrowsOnInvalidInput() {
        assertThrows(IllegalArgumentException.class, () -> {
            functionUnderTest(invalidInput);
        });
    }
}''',

            "javascript": '''describe('functionUnderTest', () => {
    test('returns expected value for valid input', () => {
        const result = functionUnderTest(inputValue);
        expect(result).toBe(expectedValue);
    });
    
    test('handles empty input', () => {
        const result = functionUnderTest([]);
        expect(result).toBeNull();
    });
    
    test('throws on invalid input', () => {
        expect(() => functionUnderTest(invalidInput)).toThrow();
    });
});'''
        }
        
        return examples.get(language, examples["python"])


# =============================================================================
# PROMPT BUILDER CLASS
# =============================================================================

class PromptBuilder:
    """
    Builds complete prompts for LLM API calls.
    Ensures consistency across all bot implementations.
    """
    
    def __init__(self):
        self.prompts = StandardPrompts()
    
    def build_test_creation_prompt(
        self, 
        source_code: str, 
        language: str = "python",
        module_name: str = None,
        include_example: bool = False
    ) -> tuple[str, str]:
        """
        Build prompts for test creation.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.prompts.test_creation_system_prompt(language)
        user_prompt = self.prompts.test_creation_user_prompt(source_code, language, module_name)
        
        if include_example:
            example = self.prompts.get_test_example(language)
            system_prompt += f"\n\nEXAMPLE TEST STRUCTURE:\n{example}"
        
        return system_prompt, user_prompt
    
    def build_test_repair_prompt(
        self,
        broken_code: str,
        error_message: str,
        error_type: str
    ) -> tuple[str, str]:
        """
        Build prompts for test repair.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.prompts.test_repair_system_prompt()
        user_prompt = self.prompts.test_repair_user_prompt(
            broken_code, error_message, error_type
        )
        
        return system_prompt, user_prompt
    
    def build_test_suite_prompt(
        self,
        test_files: list,
        language: str = "python"
    ) -> tuple[str, str]:
        """
        Build prompts for test suite creation.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.prompts.test_suite_system_prompt(language)
        user_prompt = self.prompts.test_suite_user_prompt(test_files, language)
        
        return system_prompt, user_prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prompt_hash() -> str:
    """
    Get a hash of the current prompts for version tracking.
    Useful for ensuring benchmark results used the same prompts.
    """
    import hashlib
    
    prompts = StandardPrompts()
    content = (
        prompts.test_creation_system_prompt("python") +
        prompts.test_repair_system_prompt() +
        prompts.test_suite_system_prompt("python")
    )
    
    return hashlib.md5(content.encode()).hexdigest()[:8]


def print_prompts():
    """Print all prompts for review."""
    prompts = StandardPrompts()
    
    print("=" * 70)
    print("STANDARDIZED PROMPTS FOR AI TESTING BOT BENCHMARK")
    print(f"Prompt Version Hash: {get_prompt_hash()}")
    print("=" * 70)
    
    print("\n### TEST CREATION SYSTEM PROMPT (Python) ###")
    print("-" * 50)
    print(prompts.test_creation_system_prompt("python"))
    
    print("\n### TEST CREATION USER PROMPT ###")
    print("-" * 50)
    print(prompts.test_creation_user_prompt("# [SOURCE CODE HERE]", "python"))
    
    print("\n### TEST REPAIR SYSTEM PROMPT ###")
    print("-" * 50)
    print(prompts.test_repair_system_prompt())
    
    print("\n### TEST REPAIR USER PROMPT ###")
    print("-" * 50)
    print(prompts.test_repair_user_prompt("# [BROKEN CODE]", "[ERROR MESSAGE]", "syntax_error"))


if __name__ == "__main__":
    print_prompts()