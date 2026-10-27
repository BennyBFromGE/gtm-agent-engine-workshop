"""Offline graph tests for signed-in rep identity."""

import json
import os
import re
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SignedInRepTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def run_graph(self, responder, message, user_id):
        requests = []

        def generate(model, messages, **kwargs):
            requests.append(messages)
            return ChatResult(generations=[ChatGeneration(message=responder(messages))])

        with patch.object(ChatOpenAI, "_generate", generate):
            result = agent.run_agent(message, user_id=user_id)
        return result, requests

    def test_email_has_identity_on_first_turn_and_uses_default_sender(self):
        for rep in agent.REP_IDS[:2]:
            with self.subTest(rep=rep["rep_id"]):

                def respond(messages):
                    identity = re.search(
                        r"The signed-in rep is (.*?) \((.*?)\);", messages[0].text
                    )
                    self.assertIsNotNone(identity)
                    self.assertEqual(identity.groups(), (rep["name"], rep["email"]))
                    results = [m for m in messages if isinstance(m, ToolMessage)]
                    if not results:
                        name, args = "get_prospect", {"prospect_id": "LEAD-12853"}
                    elif results[-1].name == "get_prospect":
                        name, args = (
                            "send_prospect_email",
                            {
                                "prospect": json.loads(results[-1].content)["prospect"],
                                "subject": "Demo",
                                "body": f"Let's schedule a demo.\n\n{identity[1]}",
                            },
                        )
                    else:
                        return AIMessage(content="Email sent.")
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": name,
                                "args": args,
                                "id": str(len(results)),
                            }
                        ],
                    )

                result, requests = self.run_graph(
                    respond, "Email LEAD-12853 to schedule a demo.", rep["rep_id"]
                )
                self.assertEqual(len(requests), 3)
                tools = [m for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertEqual(
                    [m.name for m in tools], ["get_prospect", "send_prospect_email"]
                )
                sent = json.loads(tools[-1].content)
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], rep["email"])
                self.assertEqual(sent["from_name"], rep["name"])
                self.assertTrue(sent["body"].endswith(rep["name"]))

    def test_scoring_needs_no_identity_tool(self):
        score = agent.ProspectScore(
            score=50,
            justification="Partial fit.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=50,
                tech_stack_match=50,
                segment_fit=50,
            ),
        )

        def respond(messages):
            results = {
                m.name: json.loads(m.content)
                for m in messages
                if isinstance(m, ToolMessage)
            }
            if not results:
                calls = [
                    {
                        "name": "build_prospect_profile",
                        "args": {"prospect_id": "LEAD-71001"},
                        "id": "profile",
                    },
                    {
                        "name": "lookup_offering",
                        "args": {"offering_id": "OFFER-10005"},
                        "id": "offering",
                    },
                ]
            elif "score_prospect" not in results:
                calls = [
                    {
                        "name": "score_prospect",
                        "args": {
                            "prospect_profile": results["build_prospect_profile"][
                                "prospect_profile"
                            ],
                            "offering": results["lookup_offering"]["offering"],
                        },
                        "id": "score",
                    }
                ]
            else:
                return AIMessage(content=str(results["score_prospect"]["score"]))
            return AIMessage(content="", tool_calls=calls)

        with patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = score
            result, _ = self.run_graph(
                respond, "Score LEAD-71001 against OFFER-10005.", "rep_jchen"
            )
        self.assertEqual(result["reply"], "50.0")
        self.assertNotIn(
            "get_current_rep",
            [m.name for m in result["messages"] if isinstance(m, ToolMessage)],
        )

    def test_rep_lookup_remains_available(self):
        def respond(messages):
            if isinstance(messages[-1], ToolMessage):
                return AIMessage(content=messages[-1].content)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_current_rep",
                        "args": {},
                        "id": "rep",
                    }
                ],
            )

        for user_id in ("rep_jchen", "unknown"):
            with self.subTest(user_id=user_id):
                result, _ = self.run_graph(respond, "Look up my rep details.", user_id)
                rep = agent.data_service.get_rep(user_id)
                self.assertEqual(
                    json.loads(result["reply"]), {"rep": rep, "found": rep is not None}
                )


if __name__ == "__main__":
    unittest.main()
