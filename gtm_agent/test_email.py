import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langsmith import tracing_context

with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-key",
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent.gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_tool_documents_plain_text_limitation(self):
        self.assertIn("plain-text email only", send_prospect_email.description)
        self.assertIn("cannot attach files", send_prospect_email.description)

    def test_plain_text_email_sends(self):
        for body in (
            "Are you available for a demo next week?",
            "The pricing deck is not attached and must be sent separately.",
        ):
            with self.subTest(body=body):
                result = send_prospect_email.func(
                    prospect={"name": "Sofia Rossi", "email": "sofia@example.com"},
                    subject="Following up",
                    body=body,
                    runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_sbrown"}}),
                )
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["body"], body)
                self.assertEqual(result["to"], "sofia@example.com")
                self.assertEqual(result["from"], "sara.brown@northpoint.com")
                self.assertTrue(result["message_id"])

    @unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires OPENAI_API_KEY")
    def test_pricing_deck_request_end_to_end(self):
        with tracing_context(enabled=False):
            result = run_agent(
                "Email LEAD-50003 to send over our pricing deck.",
                user_id="rep_sbrown",
            )
        sent = [
            json.loads(message.content)
            for message in result["messages"]
            if message.type == "tool" and message.name == "send_prospect_email"
        ]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["status"], "sent")
        self.assertEqual(sent[0]["to"], "sofia.rossi@greenfieldnetworks.com")
        reply = result["reply"].lower()
        self.assertRegex(reply, r"(?:separately|separate email|separate message)")
        self.assertRegex(reply, r"(?:cannot|can't|unable|not included|not attached|wasn't|couldn't)")
        for text in (sent[0]["subject"], sent[0]["body"], result["reply"]):
            with self.subTest(text=text):
                self.assertNotRegex(
                    text.lower(),
                    r"(?:i(?:'ve| have)?|we(?:'ve| have)?) (?:attached|delivered|shared)"
                    r"|(?:attached|enclosed) (?:is|you(?:'ll| will) find)"
                    r"|(?:deck|document|file) (?:is|was|has been) (?:attached|delivered|sent|included)"
                    r"|(?:sent|emailed|shared|delivered) (?:you |the prospect )?(?:our |the |a )?(?:pricing )?(?:deck|document|file)"
                    r"|https?://",
                )


if __name__ == "__main__":
    unittest.main()
