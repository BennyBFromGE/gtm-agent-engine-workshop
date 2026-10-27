import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent.gtm_agent import get_prospect, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})
        self.rep = {"name": "Sales Rep", "email": "rep@example.com"}

    def send(self, prospect):
        return send_prospect_email.func(
            prospect, "Introduction", "Hello", self.runtime, from_rep=self.rep
        )

    def test_disqualified_record_blocks_message_creation(self):
        for number in range(50001, 50006):
            with self.subTest(prospect_id=number):
                prospect = get_prospect.invoke({"prospect_id": f"LEAD-{number}"})["prospect"]
                record_email = prospect["email"]
                prospect = {**prospect, "disqualified": False, "email": "other@example.com"}
                with patch("gtm_agent.gtm_agent.uuid.uuid4") as message_id:
                    result = self.send(prospect)
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; email not sent.",
                    "to": record_email,
                })
                message_id.assert_not_called()

    def test_eligible_prospect_is_sent(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
        self.assertIsNotNone(prospect)
        self.assert_sent(prospect)

    def test_missing_id_or_record_preserves_send(self):
        for prospect in (
            {"name": "Prospect", "email": "prospect@example.com"},
            {"prospect_id": "missing", "name": "Prospect", "email": "prospect@example.com"},
        ):
            with self.subTest(prospect=prospect):
                self.assert_sent(prospect)

    def test_missing_email_preserves_failure(self):
        self.assertEqual(self.send({}), {
            "status": "failed", "error": "Prospect record has no email address."
        })

    def assert_sent(self, prospect):
        result = self.send(prospect)
        self.assertTrue(result.pop("message_id").startswith("msg-"))
        self.assertEqual(result, {
            "status": "sent", "to": prospect["email"], "to_name": prospect.get("name"),
            "from": self.rep["email"], "from_name": self.rep["name"],
            "subject": "Introduction", "body": "Hello",
        })


if __name__ == "__main__":
    unittest.main()
