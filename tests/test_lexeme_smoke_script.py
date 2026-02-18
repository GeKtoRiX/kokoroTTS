from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_english_lexemes.py"


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


def test_check_english_lexemes_help_has_no_verify_flag():
    result = _run_script(["--help"])
    assert result.returncode == 0
    assert "--verify-llm" not in result.stdout
