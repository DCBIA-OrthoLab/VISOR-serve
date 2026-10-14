"""scripts/setup-server.sh: the helpers that handle what an operator typed.

The functions are lifted out of the script and run by `sh`, so what is tested
is the shell code itself, without running sg, docker or the install.
"""

import os
import re
import shutil
import subprocess

import pytest

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "scripts", "setup-server.sh")

pytestmark = pytest.mark.skipif(not os.path.isfile(_SCRIPT) or shutil.which("sh") is None,
                                reason="scripts/setup-server.sh or sh is not available here")


def _function(name: str) -> str:
    text = open(_SCRIPT, encoding="utf-8").read()
    match = re.search(rf"^{name}\(\) {{\n.*?^}}\n", text, re.S | re.M)
    assert match, f"{name} is not defined in setup-server.sh"
    return match.group(0)


def _sh(body: str, *args) -> str:
    done = subprocess.run(["sh", "-c", body, "sh"] + list(args), capture_output=True, text=True, check=True)
    return done.stdout


HOSTILE = ["--token", "ab$(echo X)cd", "it's", "a`echo Y`b", "$HOME", 'x"y', "a\\b"]


def test_the_re_run_command_hands_every_argument_back_verbatim():
    """What `sg docker -c` is given is read once, by one shell: an argument
    holding $(...) or backquotes must arrive as text, not run."""
    body = _function("shell_quote") + '''
ORIGINAL_ARGS=""
for _arg in "$@"; do ORIGINAL_ARGS="$ORIGINAL_ARGS $(shell_quote "$_arg")"; done
sh -c "printf '[%s]\\n' $ORIGINAL_ARGS"
'''
    assert _sh(body, *HOSTILE).splitlines() == [f"[{arg}]" for arg in HOSTILE]


def test_the_re_run_uses_no_eval():
    text = open(_SCRIPT, encoding="utf-8").read()
    assert not re.search(r"^\s*eval .*sg docker", text, re.M)
