"""An owned Doctor can restart immediately, but never adopts another listener."""
import socket
import sys

import pytest

from src.online_rl.tree.idle_eval_worker import DedicatedDoctor


def doctor(tmp_path, monkeypatch, port):
    monkeypatch.setenv('SHERPA_OLLAMA_MODELS', str(tmp_path / 'missing-model-cache'))
    return DedicatedDoctor({'five_gpu': {'doctor_urls': []}}, tmp_path, 4,
                           sys.executable, port)


def test_owned_doctor_port_is_reusable_during_time_wait(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with socket.create_connection(('127.0.0.1', port)) as client:
            server, _ = listener.accept()
            # The server initiates close, leaving its local endpoint TIME_WAIT.
            server.close()
            assert client.recv(1) == b''
    # Passing the port check reaches the next preflight without starting a model.
    with pytest.raises(FileNotFoundError, match='model cache'):
        doctor(tmp_path, monkeypatch, port).start(None, None)


def test_live_listener_still_blocks_doctor_start(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            doctor(tmp_path, monkeypatch, port).start(None, None)
