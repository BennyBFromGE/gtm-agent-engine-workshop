import importlib
import json
import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ProspectScoringTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.object(agent.data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = "OFFER-10001"
        self.args = {
            "prospect_id": self.prospect_id,
            "offering_id": self.offering_id,
        }
        self.result = agent.ProspectScore(
            score=72,
            justification="Revenue and technology fit.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=60, segment_fit=75
            ),
        )
        self.llm = Mock()
        self.llm.invoke.return_value = self.result
        self.scoring = patch.object(agent, "_scoring_llm", self.llm)
        self.scoring.start()
        self.addCleanup(self.scoring.stop)

    def test_build_profile_preserves_result_and_cache(self):
        record = agent.data_service.get_prospect_record(self.prospect_id)
        expected = {"prospect_id": self.prospect_id, **record}
        self.assertEqual(agent._build_profile(self.prospect_id), expected)
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}),
            {"prospect_profile": expected, "found": True},
        )
        self.assertEqual(agent.data_service._PROFILES[self.prospect_id], expected)

    def test_build_profile_missing(self):
        self.assertIsNone(agent._build_profile("missing"))
        self.assertEqual(
            agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
            {"prospect_profile": None, "found": False},
        )

    def test_score_loads_records_and_preserves_scoring_input(self):
        result = agent.score_prospect.invoke(self.args)
        self.assertEqual(result, self.result.model_dump())
        expected_profile = {
            "prospect_id": self.prospect_id,
            **agent.data_service.get_prospect_record(self.prospect_id),
        }
        self.assertEqual(
            self.llm.invoke.call_args.args[0],
            [
                {"role": "system", "content": agent.SCORING_PROMPT},
                {
                    "role": "user",
                    "content": "Offering:\n"
                    + json.dumps(agent.data_service.get_offering(self.offering_id), indent=2)
                    + "\n\nProspect profile:\n"
                    + json.dumps(expected_profile, indent=2),
                },
            ],
        )

    def test_score_refreshes_cached_tech_stack(self):
        profile = agent._build_profile(self.prospect_id)
        agent.data_service._PROFILES[self.prospect_id] = {
            **profile, "tech_stack": ["stale"]
        }
        self.assertEqual(agent.score_prospect.invoke(self.args), self.result.model_dump())
        content = self.llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile, profile)

    def test_score_missing_profile(self):
        self.assertEqual(
            agent.score_prospect.invoke({**self.args, "prospect_id": "missing"}),
            {"score": None, "error": "Cannot score without a valid prospect profile."},
        )
        self.llm.invoke.assert_not_called()

    def test_score_missing_offering(self):
        self.assertEqual(
            agent.score_prospect.invoke({**self.args, "offering_id": "missing"}),
            {"score": None, "error": "Cannot score without a valid offering."},
        )
        self.llm.invoke.assert_not_called()

    def test_score_missing_or_invalid_offering(self):
        offering = agent.data_service.get_offering(self.offering_id)
        invalid_offerings = [None, {}]
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            invalid_offerings.append({k: v for k, v in offering.items() if k != field})
        for invalid in invalid_offerings:
            with self.subTest(offering=invalid):
                with patch.object(agent.data_service, "get_offering", return_value=invalid):
                    self.assertEqual(
                        agent.score_prospect.invoke(self.args),
                        {"score": None, "error": "Cannot score without a valid offering."},
                    )
        self.llm.invoke.assert_not_called()

    def test_agent_executes_id_only_scoring_call(self):
        schema = agent.score_prospect.tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"prospect_id", "offering_id"})
        self.assertEqual(set(schema["required"]), {"prospect_id", "offering_id"})
        model = ToolCallingModel(messages=iter([
            AIMessage(content="", tool_calls=[{
                "name": "score_prospect", "args": self.args, "id": "score-1"
            }]),
            AIMessage(content="The prospect scored 72."),
        ]))
        graph = agent.create_deep_agent(model=model, tools=[agent.score_prospect])
        result = graph.invoke({"messages": [{
            "role": "user",
            "content": f"Score {self.prospect_id} for {self.offering_id}",
        }]})
        tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(json.loads(tool_results[0].content), self.result.model_dump())
        self.assertEqual(result["messages"][-1].content, "The prospect scored 72.")


if __name__ == "__main__":
    unittest.main()
