#!/usr/bin/env python3
"""
LLM Bot Implementations for AI Testing Bot Benchmark Framework

All bots use IDENTICAL prompts from standard_prompts.py to ensure fair comparison.

Supported LLMs:
- ChatGPT (OpenAI GPT-4/GPT-4o)
- Claude (Anthropic)
- Google Gemini

Setup:
    pip install openai anthropic google-generativeai python-dotenv --break-system-packages
    
    Create a .env file with:
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    GOOGLE_API_KEY=AIza...
"""

import os
import re
from abc import ABC, abstractmethod

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use environment variables

from standard_prompts import PromptBuilder, get_prompt_hash

# Import API clients
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
except ImportError:
    genai = None


class BotInterface(ABC):
    """
    Abstract base class for AI testing bots.
    All implementations use the same prompts from PromptBuilder.
    """
    
    def __init__(self, name: str, api_key: str = None):
        self.name = name
        self.api_key = api_key
        self.total_tokens_used = 0
        self.total_api_calls = 0
        self.prompt_builder = PromptBuilder()
        self.prompt_version = get_prompt_hash()
    
    @abstractmethod
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call to the LLM. Implemented by each bot."""
        pass
    
    def generate_tests(self, source_code: str, module_name: str = None, language: str = "python") -> str:
        """Generate tests using standardized prompts."""
        system_prompt, user_prompt = self.prompt_builder.build_test_creation_prompt(
            source_code=source_code,
            language=language,
            module_name=module_name,
            include_example=False
        )
        response = self._call_api(system_prompt, user_prompt)
        return self._extract_code(response, language)
    
    def repair_test(self, broken_code: str, error_message: str, error_type: str) -> str:
        """Repair broken test using standardized prompts."""
        system_prompt, user_prompt = self.prompt_builder.build_test_repair_prompt(
            broken_code=broken_code,
            error_message=error_message,
            error_type=error_type
        )
        response = self._call_api(system_prompt, user_prompt)
        return self._extract_code(response)
    
    def _extract_code(self, response: str, language: str = None) -> str:
        """Extract code from response, removing any markdown formatting."""
        if not response:
            return ""
        
        patterns = [
            r'```(?:python|java|javascript|js|typescript|ts|json)?\n?(.*?)```',
            r'```\n?(.*?)```',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        return response.strip()
    
    def get_info(self) -> dict:
        """Get bot information for logging."""
        return {
            "name": self.name,
            "prompt_version": self.prompt_version,
            "api_calls": self.total_api_calls,
            "tokens_used": self.total_tokens_used
        }


class ChatGPTBot(BotInterface):
    """ChatGPT implementation using OpenAI API."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gpt-4.1",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        debug: bool = False
    ):
        super().__init__("ChatGPT")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.debug:
                print(f"\n[DEBUG ChatGPT] === REQUEST ===")
                print(f"[DEBUG ChatGPT] System prompt:\n{system_prompt[:500]}...")
                print(f"[DEBUG ChatGPT] User prompt:\n{user_prompt[:500]}...")
            
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=user_prompt
            )
            self.total_api_calls += 1
            output = response.output_text
            
            if self.debug:
                print(f"\n[DEBUG ChatGPT] === RESPONSE ===")
                print(f"[DEBUG ChatGPT] Output ({len(output)} chars):\n{output[:1000]}...")
                if len(output) > 1000:
                    print(f"[DEBUG ChatGPT] ... (truncated, full length: {len(output)})")
            
            return output
        except Exception as e:
            print(f"[ChatGPT] API error: {e}")
            return ""


class ClaudeBot(BotInterface):
    """Claude implementation using Anthropic API."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        debug: bool = False
    ):
        super().__init__("Claude")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY env var.")
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.debug:
                print(f"\n[DEBUG Claude] === REQUEST ===")
                print(f"[DEBUG Claude] System prompt:\n{system_prompt[:500]}...")
                print(f"[DEBUG Claude] User prompt:\n{user_prompt[:500]}...")
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            self.total_api_calls += 1
            if hasattr(response, 'usage'):
                self.total_tokens_used += response.usage.input_tokens + response.usage.output_tokens
            output = response.content[0].text
            
            if self.debug:
                print(f"\n[DEBUG Claude] === RESPONSE ===")
                print(f"[DEBUG Claude] Output ({len(output)} chars):\n{output[:1000]}...")
                if len(output) > 1000:
                    print(f"[DEBUG Claude] ... (truncated, full length: {len(output)})")
            
            return output
        except Exception as e:
            print(f"[Claude] API error: {e}")
            return ""


class GeminiBot(BotInterface):
    """Google Gemini implementation using Generative AI API."""
    
    def __init__(
        self, 
        api_key: str = None, 
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        debug: bool = False
    ):
        super().__init__("Gemini")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug
        
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var.")
        if genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-genai")
        
        self.client = genai.Client(api_key=self.api_key)
    
    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        try:
            if self.debug:
                print(f"\n[DEBUG Gemini] === REQUEST ===")
                print(f"[DEBUG Gemini] System prompt:\n{system_prompt[:500]}...")
                print(f"[DEBUG Gemini] User prompt:\n{user_prompt[:500]}...")
            
            response = self.client.models.generate_content(
                model=self.model_name,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt
                ),
                contents=user_prompt
            )
            self.total_api_calls += 1
            output = response.text
            
            if self.debug:
                print(f"\n[DEBUG Gemini] === RESPONSE ===")
                print(f"[DEBUG Gemini] Output ({len(output)} chars):\n{output[:1000]}...")
                if len(output) > 1000:
                    print(f"[DEBUG Gemini] ... (truncated, full length: {len(output)})")
            
            return output
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return ""


def create_bot(bot_name: str, api_key: str = None, debug: bool = False, **kwargs) -> BotInterface:
    """
    Create a bot instance by name.
    
    Args:
        bot_name: "chatgpt", "claude", or "gemini"
        api_key: Optional API key (uses env var if not provided)
        debug: Enable debug output for API calls
        **kwargs: Additional args (model, temperature, max_tokens)
    """
    bot_name = bot_name.lower().strip()
    
    bot_map = {
        "chatgpt": ChatGPTBot,
        "gpt": ChatGPTBot,
        "openai": ChatGPTBot,
        "claude": ClaudeBot,
        "anthropic": ClaudeBot,
        "gemini": GeminiBot,
        "google": GeminiBot,
    }
    
    bot_class = bot_map.get(bot_name)
    if not bot_class:
        raise ValueError(f"Unknown bot: {bot_name}. Available: chatgpt, claude, gemini")
    
    return bot_class(api_key=api_key, debug=debug, **kwargs)


def get_available_bots() -> list:
    """Get list of bots that have API keys configured."""
    available = []
    checks = [
        ("chatgpt", "OPENAI_API_KEY", "ChatGPT"),
        ("claude", "ANTHROPIC_API_KEY", "Claude"),
        ("gemini", "GOOGLE_API_KEY", "Gemini"),
    ]
    for bot_id, env_var, display_name in checks:
        if os.getenv(env_var):
            available.append((bot_id, display_name))
    return available


if __name__ == "__main__":
    print("LLM Bots: ChatGPT, Claude, Gemini")
    print(f"Prompt Version: {get_prompt_hash()}")
    print("\nAPI Key Status:")
    for name, env_var in [("ChatGPT", "OPENAI_API_KEY"), ("Claude", "ANTHROPIC_API_KEY"), ("Gemini", "GOOGLE_API_KEY")]:
        status = "✓ Ready" if os.getenv(env_var) else "✗ Not set"
        print(f"  {name}: {status}")