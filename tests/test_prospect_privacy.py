import copy
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service, gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-12853"
        self.record = copy.deepcopy(data_service.PROSPECTS[self.prospect_id])
        self.records = patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record})
        self.records.start()
        self.addCleanup(self.records.stop)
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)

    def expected_profile(self):
        return {
            "prospect_id": self.prospect_id,
            **{k: v for k, v in self.record.items()
               if k not in ("billing_qualification", "password", "tax_id")},
        }

    def test_contact_allowlist_and_email_flow(self):
        for with_billing in (True, False):
            with self.subTest(with_billing=with_billing):
                if not with_billing:
                    self.record.pop("billing_qualification")
                self.record["password"] = "private-credential"
                contact = gtm_agent.get_prospect.invoke({"prospect_id": self.prospect_id})
                expected = {
                    "prospect_id": self.prospect_id,
                    "name": self.record["name"],
                    "email": self.record["email"],
                    "disqualified": self.record["disqualified"],
                }
                self.assertEqual(contact, {"prospect": expected, "found": True})
                sent = gtm_agent.send_prospect_email.func(
                    prospect=contact["prospect"], subject="Hello", body="Let's talk",
                    runtime=SimpleNamespace(config={}),
                    from_rep={"name": "Rep", "email": "rep@example.com"},
                )
                self.assertEqual(sent["to"], self.record["email"])
                self.assertEqual(sent["to_name"], self.record["name"])
                self.assertNotIn("private-credential", json.dumps(sent))

    def test_new_profiles_preserve_enrichment_without_sensitive_fields(self):
        for with_billing in (True, False):
            with self.subTest(with_billing=with_billing):
                data_service._PROFILES.clear()
                if not with_billing:
                    self.record.pop("billing_qualification")
                self.record.update(password="private-credential", tax_id="private-tax-id")
                built = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
                expected = {"prospect_profile": self.expected_profile(), "found": True}
                self.assertEqual(built, expected)
                self.assertEqual(data_service._PROFILES[self.prospect_id], self.expected_profile())
                self.assertEqual(gtm_agent.build_prospect_profile.invoke(
                    {"prospect_id": self.prospect_id}), expected)
                if with_billing:
                    self.assertIn("billing_qualification", self.record)

    def test_stale_cache_is_clean_at_traced_return_boundary(self):
        data_service._PROFILES[self.prospect_id] = {
            "prospect_id": self.prospect_id, **copy.deepcopy(self.record),
            "password": "private-credential",
        }
        runs = []
        with tracing_context(enabled="local"):
            result = data_service.get_profile_from_db(
                self.prospect_id, langsmith_extra={"on_end": runs.append})
        expected = {"prospect_profile": self.expected_profile()}
        self.assertEqual(result, expected)
        self.assertEqual(data_service._PROFILES[self.prospect_id], self.expected_profile())
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].outputs, expected)
        self.assertEqual(gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}), {**expected, "found": True})

    def test_scoring_payload_allowlist_and_saved_tech_stack(self):
        scoring_result = {"score": 75, "justification": "Good fit"}
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = scoring_result
        offering = data_service.OFFERINGS["OFFER-10001"]
        for with_billing in (True, False):
            with self.subTest(with_billing=with_billing):
                data_service._PROFILES.clear()
                if not with_billing:
                    self.record.pop("billing_qualification")
                profile = gtm_agent.build_prospect_profile.invoke(
                    {"prospect_id": self.prospect_id})["prospect_profile"]
                self.assertEqual(profile, self.expected_profile())
                profile = {**profile, "tech_stack": ["Stale technology"]}
                if with_billing:
                    profile["billing_qualification"] = self.record["billing_qualification"]
                with patch.object(gtm_agent, "_scoring_llm", llm):
                    result = gtm_agent.score_prospect.invoke(
                        {"prospect_profile": profile, "offering": offering})
                self.assertEqual(result, scoring_result)
                prompt = llm.invoke.call_args.args[0][1]["content"]
                payload = json.loads(prompt.split("\n\nProspect profile:\n", 1)[1])
                self.assertEqual(payload, {
                    "prospect_id": self.prospect_id,
                    **{k: self.record[k] for k in (
                        "name", "annual_revenue", "tech_stack", "account_details", "engagement_history")},
                })

    def test_all_prospects_flow_to_scoring_without_billing(self):
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = {"score": 75}
        with patch.object(gtm_agent, "_scoring_llm", llm):
            for prospect_id, record in data_service.PROSPECTS.items():
                with self.subTest(prospect_id=prospect_id):
                    contact = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})
                    profile = gtm_agent.build_prospect_profile.invoke(
                        {"prospect_id": prospect_id})["prospect_profile"]
                    scoring_args = {
                        "prospect_profile": profile,
                        "offering": data_service.OFFERINGS["OFFER-10001"],
                    }
                    gtm_agent.score_prospect.invoke(scoring_args)
                    exposed = json.dumps({
                        "contact": contact, "scoring_args": scoring_args,
                        "prompt": llm.invoke.call_args.args[0],
                        "cached": data_service.get_profile_from_db(prospect_id),
                    })
                    self.assertNotIn("billing_qualification", exposed)
                    for value in record["billing_qualification"].values():
                        self.assertNotIn(value, exposed)

    def test_missing_records_and_optional_disqualified(self):
        self.record.pop("disqualified")
        contact = gtm_agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(contact["prospect"], {
            "prospect_id": self.prospect_id, "name": self.record["name"],
            "email": self.record["email"],
        })
        self.assertEqual(gtm_agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(gtm_agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})


if __name__ == "__main__":
    unittest.main()
