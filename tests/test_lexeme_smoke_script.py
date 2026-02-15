import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_english_lexemes.py"


class _MockLmStudioHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        request_payload = json.loads(body)

        messages = request_payload.get("messages", [])
        user_content = messages[-1].get("content", "") if messages else ""
        if isinstance(user_content, str) and user_content.startswith("/no_think"):
            user_content = user_content.split("\n", 1)[1] if "\n" in user_content else "{}"
        user_payload = json.loads(user_content) if isinstance(user_content, str) else {}
        tokens = user_payload.get("tokens", [])

        token_checks = []
        for token in tokens:
            token_checks.append(
                {
                    "token_index": int(token.get("token_index", 0)),
                    "lemma": str(token.get("lemma", "")),
                    "upos": str(token.get("upos", "X")).upper(),
                    "feats": token.get("feats", {}) if isinstance(token.get("feats"), dict) else {},
                }
            )

        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"token_checks": token_checks, "new_expressions": []}
                        )
                    }
                }
            ]
        }
        response_raw = json.dumps(response_payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_raw)))
        self.end_headers()
        self.wfile.write(response_raw)

    def log_message(self, _format, *_args):
        return


def _run_script(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT_PATH)] + args
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_english_lexemes_script_runs_successfully():
    result = _run_script(["--text", "We're 3.14% ready for tests.", "--json"])
    assert result.returncode == 0
    assert "LEXEME_CHECK_OK" in result.stdout
    assert '"token_count"' in result.stdout


def test_check_english_lexemes_script_runs_llm_verify():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockLmStudioHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = os.environ.copy()
        env["LM_VERIFY_BASE_URL"] = f"http://127.0.0.1:{server.server_port}/v1"
        env["LM_VERIFY_API_KEY"] = "lm-studio"
        env["LM_VERIFY_MODEL"] = "test-model"
        env["LM_VERIFY_TIMEOUT_SECONDS"] = "5"
        env["LM_VERIFY_TEMPERATURE"] = "0"
        env["LM_VERIFY_MAX_TOKENS"] = "256"

        result = _run_script(
            ["--text", "Look up the word now.", "--verify-llm", "--json"],
            env=env,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert result.returncode == 0
    assert "LEXEME_CHECK_OK" in result.stdout
    assert "LLM_VERIFY_OK" in result.stdout
    assert '"mismatch_count": 0' in result.stdout
