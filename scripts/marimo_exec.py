#!/usr/bin/env python3
"""
Marimo Kernel Remote Code Execution Script.
Communicates with Marimo's HTTP API (/api/sessions and /api/kernel/execute)
to run Python code inside an active notebook kernel.
"""

import os
import sys
import json
import argparse
import urllib.request
from typing import Optional, Tuple

def get_session_id(base_url: str, token: str) -> str:
    api_url = f"{base_url.rstrip('/')}/api/sessions"
    req = urllib.request.Request(
        api_url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        if not data:
            raise RuntimeError("No active session found on Marimo server!")
        # Return first active session id
        return list(data.keys())[0]

def execute_remote(
    base_url: str,
    token: str,
    code: str,
    session_id: Optional[str] = None,
    timeout: int = 3600
) -> Tuple[bool, str]:
    if not session_id:
        session_id = get_session_id(base_url, token)
    
    api_url = f"{base_url.rstrip('/')}/api/kernel/execute"
    payload = json.dumps({"code": code}).encode("utf-8")
    
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Marimo-Session-Id": session_id,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        },
        method="POST"
    )
    
    output_parts = []
    current_event = None
    success = True
    
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                current_event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                try:
                    payload_data = json.loads(data_str)
                    if current_event in ("stdout", "stderr"):
                        text = payload_data.get("data", "")
                        if current_event == "stdout":
                            sys.stdout.write(text)
                            sys.stdout.flush()
                        else:
                            sys.stderr.write(text)
                            sys.stderr.flush()
                        output_parts.append(text)
                    elif current_event == "done":
                        success = payload_data.get("success", True)
                        if not success:
                            err = payload_data.get("error", {})
                            msg = err.get("msg", "Unknown Marimo execution error")
                            sys.stderr.write(f"\n[Marimo Execution Error]: {msg}\n")
                            output_parts.append(f"\nError: {msg}\n")
                        break
                except json.JSONDecodeError:
                    pass

    return success, "".join(output_parts)

def main():
    parser = argparse.ArgumentParser(description="Execute Python code inside remote Marimo kernel.")
    parser.add_argument("--url", type=str, required=True, help="Marimo notebook URL")
    parser.add_argument("--token", type=str, required=True, help="Auth token")
    parser.add_argument("--session_id", type=str, default=None, help="Marimo session ID (auto-detected if omitted)")
    parser.add_argument("--code", type=str, default=None, help="Inline code to execute")
    parser.add_argument("--file", type=str, default=None, help="File containing code to execute")
    parser.add_argument("--timeout", type=int, default=3600, help="Execution timeout in seconds (default: 3600)")
    
    args = parser.parse_args()
    
    if args.code is not None:
        code_str = args.code
    elif args.file is not None:
        with open(args.file, "r", encoding="utf-8") as f:
            code_str = f.read()
    elif not sys.stdin.isatty():
        code_str = sys.stdin.read()
    else:
        # Default ping
        code_str = 'import sys; print(f"Marimo Connected! Python {sys.version}")'
        
    ok, out = execute_remote(args.url, args.token, code_str, args.session_id, timeout=args.timeout)
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
