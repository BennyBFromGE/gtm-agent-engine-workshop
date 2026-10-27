import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class AgentTests(unittest.TestCase):
    def test_email_workflow_context_and_runtime_injection(self):
        expected_tools = {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info",
            "get_current_rep",
        }
        prospect = agent.data_service.get_prospect_record("LEAD-71001")
        for rep_id in ("rep_amills", "rep_jchen"):
            with self.subTest(rep_id=rep_id):
                calls = []
                replies = iter([
                    AIMessage(content="", tool_calls=[{
                        "name": "get_current_rep", "args": {}, "id": "rep",
                    }]),
                    AIMessage(content="", tool_calls=[{
                        "name": "get_prospect", "args": {"prospect_id": "LEAD-71001"},
                        "id": "prospect",
                    }, {
                        "name": "lookup_offering", "args": {"offering_id": "OFFER-10001"},
                        "id": "offering",
                    }]),
                    AIMessage(content="", tool_calls=[{
                        "name": "send_prospect_email", "args": {
                            "prospect": {"name": prospect["name"], "email": prospect["email"]},
                            "subject": "Trial", "body": "Your trial is ready.",
                        }, "id": "email",
                    }]),
                    AIMessage(content="Email sent."),
                ])

                def respond(messages, **kwargs):
                    calls.append((messages, kwargs))
                    return ChatResult(generations=[ChatGeneration(message=next(replies))])

                with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
                        patch.object(ChatOpenAI, "_generate", side_effect=respond):
                    result = agent.run_agent(
                        "Email LEAD-71001 about a trial of OFFER-10001.", user_id=rep_id,
                        thread_id="test-thread", environment="test",
                    )

                self.assertEqual(result["reply"], "Email sent.")
                self.assertEqual(len(calls), 4)
                for messages, kwargs in calls:
                    self.assertEqual(
                        [message.content for message in messages if message.type == "system"],
                        [agent.SYSTEM_PROMPT],
                    )
                    self.assertEqual(
                        {tool["function"]["name"] for tool in kwargs["tools"]}, expected_tools,
                    )
                    for tool in kwargs["tools"]:
                        self.assertNotIn("runtime", tool["function"]["parameters"]["properties"])
                outputs = {
                    message.tool_call_id: json.loads(message.content)
                    for message in result["messages"] if isinstance(message, ToolMessage)
                }
                rep = agent.data_service.get_rep(rep_id)
                self.assertEqual(outputs["rep"], {"rep": rep, "found": True})
                self.assertTrue(outputs["prospect"]["found"])
                self.assertTrue(outputs["offering"]["found"])
                self.assertEqual(outputs["email"]["status"], "sent")
                self.assertEqual(outputs["email"]["from"], rep["email"])
                self.assertEqual(outputs["email"]["from_name"], rep["name"])
                self.assertEqual(outputs["email"]["to"], prospect["email"])

    def test_run_agent_preserves_input_config_and_output(self):
        run_id = uuid.UUID("e986be68-c374-4fde-879a-a10842e4fdd7")
        messages = [AIMessage(content="Done.")]
        with patch.object(agent.gtm_agent, "invoke", return_value={"messages": messages}) as invoke, \
                patch.object(agent.uuid, "uuid4", return_value=run_id):
            result = agent.run_agent(
                "Email LEAD-71001", user_id="rep_amills", environment="test", thread_id="thread-1",
            )
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Email LEAD-71001"}]},
            config={
                "run_name": "GTM Assistant", "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-1", "user_id": "rep_amills",
                    "environment": "test", "request_intent": "send_email",
                },
            },
        )
        self.assertEqual(result, {"reply": "Done.", "messages": messages, "run_id": str(run_id)})


if __name__ == "__main__":
    unittest.main()
