import os
import tempfile
import unittest
from uuid import uuid4
from pathlib import Path
from unittest.mock import patch

import alexander


class ArithmeticTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(alexander.calculate("2+2*3"), "8")
        self.assertEqual(alexander.calculate("10 / 2"), "5.0")

    def test_rejects_code_and_unbounded_exponents(self):
        self.assertEqual(
            alexander.calculate("__import__('os')"),
            "I can only do basic arithmetic.",
        )
        self.assertEqual(alexander.calculate("2**100000"), "Exponent is too large.")


class PermissionTests(unittest.TestCase):
    def test_risky_tools_require_explicit_confirmation(self):
        args = {"code": "print('not run')"}
        self.assertFalse(alexander._tool_allowed("run_python", args, None))
        self.assertFalse(alexander._tool_allowed("run_python", args, lambda _: False))
        self.assertTrue(alexander._tool_allowed("run_python", args, lambda _: True))

    def test_demo_reminder_requires_confirmation(self):
        brain = alexander.Brain(force_demo=True)
        with patch("alexander.set_reminder") as set_reminder:
            reply = brain.think("remind me in 1 minute to test")
        self.assertIn("confirmation", reply)
        set_reminder.assert_not_called()

    def test_agent_does_not_run_risky_tool_without_confirmation(self):
        called = []
        schema = {
            "type": "function",
            "function": {"name": "run_python", "parameters": {}},
        }
        fake_ollama = type("FakeOllama", (), {})()
        responses = [
            {"message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {
                    "name": "run_python", "arguments": {"code": "print(1)"},
                }}],
            }},
            {"message": {
                "role": "assistant", "content": "The action was not approved.",
            }},
        ]

        def fake_chat(**_):
            return responses.pop(0)

        fake_ollama.chat = fake_chat
        agent = alexander.Agent(
            "coder", "test",
            {"run_python": {
                "schema": schema,
                "run": lambda **kwargs: called.append(kwargs),
            }},
        )
        with patch.object(alexander, "ollama", fake_ollama):
            reply = agent.run("run code")

        self.assertEqual(reply, "The action was not approved.")
        self.assertEqual(called, [])


class MemoryTests(unittest.TestCase):
    def test_memory_collections_are_scoped_by_file(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "first.json"
            second = Path(folder) / "second.json"
            self.assertEqual(
                alexander._memory_collection_name(first),
                alexander._memory_collection_name(first),
            )
            self.assertNotEqual(
                alexander._memory_collection_name(first),
                alexander._memory_collection_name(second),
            )

    def test_corrupt_memory_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "memory.json"
            original = "{invalid json"
            path.write_text(original, encoding="utf-8")
            memory = alexander.Memory(path=str(path))
            with self.assertRaises(RuntimeError):
                memory.remember("important fact")
            self.assertEqual(path.read_text(encoding="utf-8"), original)


class PasswordTests(unittest.TestCase):
    def test_password_requires_minimum_length_and_exact_match(self):
        self.assertFalse(alexander.verify_password("short", "short"))
        self.assertTrue(alexander.verify_password("long passphrase", "long passphrase"))
        self.assertTrue(alexander.verify_password("長い-passphrase", "長い-passphrase"))
        self.assertFalse(alexander.verify_password("wrong passphrase", "long passphrase"))

    def test_password_lockout_after_repeated_failures(self):
        client_key = f"test:{uuid4().hex}"
        self.assertEqual(
            alexander.authenticate_password("wrong passphrase", None,
                                            "long passphrase"),
            (False, 0),
        )
        for _ in range(4):
            self.assertEqual(
                alexander.authenticate_password("wrong passphrase", client_key,
                                                "long passphrase"),
                (False, 0),
            )
        self.assertEqual(
            alexander.authenticate_password("wrong passphrase", client_key,
                                            "long passphrase"),
            (False, 900),
        )
        self.assertEqual(
            alexander.authenticate_password("long passphrase", client_key,
                                            "long passphrase"),
            (False, 900),
        )


class ServerAuthTests(unittest.TestCase):
    def test_server_routes_require_unlock_and_lock_revokes_session(self):
        from fastapi.testclient import TestClient

        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                with patch.dict(os.environ, {
                    "ALEXANDER_PASSWORD": "a sufficiently long passphrase",
                    "ALEXANDER_COOKIE_SECURE": "false",
                }):
                    app = alexander.create_server_app()
                    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
                        self.assertEqual(client.get("/openapi.json").status_code, 404)
                        self.assertEqual(client.get("/history").status_code, 401)
                        self.assertEqual(
                            client.post("/auth/unlock", content=b"x" * (65 * 1024)).status_code,
                            413,
                        )
                        self.assertEqual(
                            client.post(
                                "/auth/unlock",
                                json={"password": "incorrect passphrase"},
                            ).status_code,
                            401,
                        )
                        unlocked = client.post(
                            "/auth/unlock",
                            json={"password": "a sufficiently long passphrase"},
                        )
                        self.assertEqual(unlocked.status_code, 200)
                        cookie_header = unlocked.headers["set-cookie"].lower()
                        self.assertIn("httponly", cookie_header)
                        self.assertIn("samesite=strict", cookie_header)

                        with patch("alexander.handle", return_value="working"):
                            response = client.post("/ask", json={"text": "hello"})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.json(), {"reply": "working"})
                        self.assertEqual(client.get("/history").status_code, 200)

                        self.assertEqual(
                            client.post(
                                "/ask", json={"text": "hello"},
                                headers={"Origin": "http://attacker.example"},
                            ).status_code,
                            403,
                        )
                        self.assertEqual(client.post("/auth/lock").status_code, 200)
                        self.assertEqual(client.get("/history").status_code, 401)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
