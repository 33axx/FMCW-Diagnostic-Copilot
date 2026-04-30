#!/usr/bin/env python3
"""Diagnostic script — checks API key and LLM connectivity."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=== DIAGNOSTIC ===")

# 1. Check .env file
env_file = Path(__file__).resolve().parent / ".env"
print(f"\n[1] .env file: {env_file}")
print(f"    exists: {env_file.exists()}")
if env_file.exists():
    content = env_file.read_text()
    for line in content.split("\n"):
        if "KEY" in line.upper() and "=" in line and not line.strip().startswith("#"):
            key_name = line.split("=")[0].strip()
            key_val = line.split("=", 1)[1].strip()
            masked = key_val[:6] + "..." if len(key_val) > 6 else key_val
            print(f"    {key_name}={masked}")

# 2. Check env vars
print("\n[2] Environment variables:")
for var in ("MIMO_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
    val = os.environ.get(var, "")
    if val:
        print(f"    {var}={val[:6]}... (length={len(val)})")
    else:
        print(f"    {var}=<NOT SET>")

# 3. Check packages
print("\n[3] Packages:")
for pkg in ("openai", "dotenv", "numpy", "scipy"):
    try:
        __import__(pkg)
        print(f"    {pkg}: ✓")
    except ImportError:
        print(f"    {pkg}: ✗ NOT INSTALLED")

# 4. Load .env and re-check
print("\n[4] After dotenv load_dotenv:")
try:
    from dotenv import load_dotenv
    if env_file.exists():
        load_dotenv(env_file)
        for var in ("MIMO_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(var, "")
            if val:
                print(f"    {var}={val[:6]}... (length={len(val)})")
            else:
                print(f"    {var}=<NOT SET>")
    else:
        print("    .env file not found, skipping")
except ImportError:
    print("    python-dotenv not installed, skipping")

# 5. Try to resolve API key
print("\n[5] LLMConfig.resolved_api_key:")
from agents.llm_reasoner import LLMConfig
cfg = LLMConfig()
key = cfg.resolved_api_key
if key:
    print(f"    resolved: {key[:8]}... (length={len(key)})")
    print(f"    base_url: {cfg.base_url}")
    print(f"    model: {cfg.model}")
else:
    print("    resolved: <EMPTY> — THIS IS THE PROBLEM")

# 6. For DeepSeek: suggest correct base_url
print("\n[6] DeepSeek config hint:")
ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
if ds_key:
    print(f"    You have DEEPSEEK_API_KEY set.")
    print(f"    But default base_url is: {cfg.base_url}")
    print(f"    For DeepSeek, base_url should be: https://api.deepseek.com/v1")
    print(f"    Fix: set DEEPSEEK_BASE_URL=https://api.deepseek.com/v1 in .env")
    print(f"         or pass base_url to LLMConfig()")
else:
    print("    No DeepSeek key detected. Did you set DEEPSEEK_API_KEY in .env?")
