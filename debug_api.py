"""Debug script to diagnose APIConnectionError issues."""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=== Environment Variables ===")
print(f"ZAI_API_KEY: {os.getenv('ZAI_API_KEY', 'NOT SET')[:20]}...")
print(f"ZAI_ENDPOINT: {os.getenv('ZAI_ENDPOINT', 'NOT SET')}")
print(f"ZAI_MODEL: {os.getenv('ZAI_MODEL', 'NOT SET')}")
print(f"UV_NATIVE_TLS: {os.getenv('UV_NATIVE_TLS', 'NOT SET')}")

print("\n=== Testing Network Connectivity ===")
import urllib3

try:
    http = urllib3.PoolManager(
        cert_reqs="CERT_REQUIRED", ca_certs=os.environ.get("SSL_CERT_FILE")
    )
    r = http.request("GET", "https://api.z.ai/api/coding/paas/v4/", timeout=10)
    print(f"Direct HTTP request status: {r.status}")
except Exception as e:
    print(f"Direct HTTP request failed: {e}")

print("\n=== Testing OpenAI Client Directly ===")
try:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("ZAI_MODEL", "glm-5.1"),
        base_url=os.getenv("ZAI_ENDPOINT", "https://api.z.ai/api/coding/paas/v4/"),
        api_key=os.getenv("ZAI_API_KEY", ""),
        timeout=30,
    )
    result = llm.invoke("Say OK")
    print(f"OpenAI client test: SUCCESS - {result.content[:100]}")
except Exception as e:
    print(f"OpenAI client test: FAILED - {type(e).__name__}: {str(e)[:200]}")

print("\n=== Testing Wells Config ===")
try:
    from coding_harness.config import get_llm

    llm = get_llm()
    result = llm.invoke("Say OK")
    print(f"Wells config test: SUCCESS - {result.content[:100]}")
except Exception as e:
    print(f"Wells config test: FAILED - {type(e).__name__}: {str(e)[:200]}")

print("\n=== Testing Wells Runtime ===")
try:
    from coding_harness.runtime import run_step

    result, report = run_step(
        step="test",
        task_type="planning",
        system="You are a test assistant. Say OK.",
        chunks={"user_request": "Say OK"},
        workspace=os.getcwd(),
    )
    print(f"Wells runtime test: SUCCESS - {result[:100]}")
except Exception as e:
    print(f"Wells runtime test: FAILED - {type(e).__name__}: {str(e)[:200]}")

print("\n=== Testing LangGraph ===")
try:
    from coding_harness.graph import build_graph

    app = build_graph()
    initial_state = {
        "goal": "Say OK",
        "iteration": 0,
        "max_iterations": 1,
        "workspace_root": os.getcwd(),
        "safety": "auto",
        "plan_mode": False,
        "messages": [],
    }
    print("Starting LangGraph invocation (this may take 30+ seconds)...")
    result = app.invoke(initial_state, timeout=60)
    print(
        f"LangGraph test: SUCCESS - {result.get('development_plan', 'NO PLAN')[:100]}"
    )
except Exception as e:
    print(f"LangGraph test: FAILED - {type(e).__name__}: {str(e)[:200]}")

print("\n=== Diagnosis Complete ===")
