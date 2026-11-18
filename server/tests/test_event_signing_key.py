"""The run-event signing key outlives a restart."""

import os
import stat

import serve


def test_the_key_is_the_same_on_the_next_start(tmp_path):
    first = serve.event_signing_key(str(tmp_path))
    assert serve.event_signing_key(str(tmp_path)) == first


def test_the_key_file_is_readable_by_this_user_only(tmp_path):
    serve.event_signing_key(str(tmp_path))
    mode = os.stat(tmp_path / serve.EVENT_KEY_FILE).st_mode
    assert stat.S_IMODE(mode) == 0o600


def test_an_empty_key_file_is_replaced_rather_than_trusted(tmp_path):
    (tmp_path / serve.EVENT_KEY_FILE).write_text("")
    assert serve.event_signing_key(str(tmp_path))
