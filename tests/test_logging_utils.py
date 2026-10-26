import logging
import unittest

from drowning.logging_utils import RedactingFilter, redact_secrets


class LoggingRedactionTests(unittest.TestCase):
    def test_redacts_database_passwords_and_inline_tokens(self):
        value = "db=postgresql://user:verysecret@db.example/app api_key=abc123"
        redacted = redact_secrets(value)
        self.assertNotIn("verysecret", redacted)
        self.assertNotIn("abc123", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_filter_redacts_message_and_formatted_arguments(self):
        record = logging.LogRecord("test", logging.INFO, "x.py", 1,
                                   "token=%s", ("sensitive-value",), None)
        RedactingFilter().filter(record)
        self.assertNotIn("sensitive-value", record.getMessage())


if __name__ == "__main__":
    unittest.main()
