import json
import os
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}))
        self.enterContext(patch.dict(data_service.PROSPECTS, deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))
        self.prospect_id = "LEAD-71001"

    def test_new_technology_is_persisted(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Kafka", data_service.get_prospect_record(self.prospect_id)["tech_stack"])
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))

    def test_duplicate_update_keeps_one_copy(self):
        technology = data_service.fetch_tech_stack(self.prospect_id)[0]
        original = list(data_service.fetch_tech_stack(self.prospect_id))

        for _ in range(2):
            result = data_service.update_prospect_info(self.prospect_id, technology)
            self.assertTrue(result["updated"])
            self.assertEqual(result["tech_stack"], original)
            self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), original)
            self.assertEqual(result["tech_stack"].count(technology), 1)

    def test_unknown_prospect_does_not_change_records(self):
        original = deepcopy(data_service.PROSPECTS)

        result = data_service.update_prospect_info("unknown-prospect", "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertEqual(data_service.PROSPECTS, original)

    def test_cached_profile_is_rebuilt_after_update(self):
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", original["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])
        self.assertEqual(
            rebuilt["prospect_profile"]["tech_stack"],
            data_service.fetch_tech_stack(self.prospect_id),
        )

    def test_update_then_score_uses_persisted_technology(self):
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = gtm_agent.lookup_offering.invoke({"offering_id": "OFFER-10005"})["offering"]
        updated = gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Kafka",
        })
        self.assertTrue(updated["updated"])
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])

        def score_messages(messages):
            profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertIn("Kafka", profile["tech_stack"])
            missing = set(offering["required_tech_stack"]) - set(profile["tech_stack"])
            self.assertNotIn("Kafka", missing)
            return gtm_agent.ProspectScore(
                score=100,
                justification="Present: Kafka. Missing: " + ", ".join(sorted(missing)),
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_messages
            for profile in (original["prospect_profile"], rebuilt["prospect_profile"]):
                result = gtm_agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": offering,
                })
                self.assertEqual(result["justification"], "Present: Kafka. Missing: ")


if __name__ == "__main__":
    unittest.main()
