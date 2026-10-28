"""AutoGen (v0.4+ AgentChat) customer-support team using Swarm handoffs.

Master_Agent reads the user's message and hands off to exactly one department agent.
That agent answers the user, and the run ends.
"""

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Handoff
from autogen_agentchat.conditions import FunctionalTermination, MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient

DEPARTMENTS = {
    "IT_Agent": {
        "label": "IT Department",
        "icon": "💻",
        "handoff": "Delegate IT questions: laptops, software, VPN, Wi-Fi, passwords/MFA, "
                   "email access, printers, security incidents, device requests.",
        "prompt": (
            "You are the IT Department support agent of the company. Help with hardware, "
            "software, accounts, passwords/MFA, VPN, network, email and security issues. "
            "Give clear step-by-step troubleshooting. If a ticket is needed, tell the user "
            "to raise one with the IT helpdesk and list the details to include."
        ),
    },
    "HR_Agent": {
        "label": "HR Department",
        "icon": "🧑‍💼",
        "handoff": "Delegate HR questions: leave, holidays, payroll queries about policy, "
                   "onboarding, benefits, performance reviews, code of conduct, resignation.",
        "prompt": (
            "You are the HR Department support agent of the company. Help with leave "
            "policy, holidays, benefits, onboarding, performance reviews, workplace "
            "conduct and resignation process. Be empathetic and professional. For "
            "personal or confidential cases, advise contacting the HR team directly."
        ),
    },
    "Finance_Agent": {
        "label": "Finance Department",
        "icon": "💰",
        "handoff": "Delegate Finance questions: expense claims, reimbursements, invoices, "
                   "budgets, purchase orders, payment status, taxes/TDS, travel advances.",
        "prompt": (
            "You are the Finance Department support agent of the company. Help with "
            "expense claims, reimbursements, invoices, purchase orders, payment status, "
            "budgets and tax-related process questions. Explain required documents and "
            "approval steps clearly."
        ),
    },
    "Engineering_Agent": {
        "label": "Engineering Department",
        "icon": "🛠️",
        "handoff": "Delegate Engineering questions: code, deployments, CI/CD, APIs, "
                   "architecture, bugs, dev environments, repos, code review, on-call.",
        "prompt": (
            "You are the Engineering Department support agent of the company. Help with "
            "development environments, repos, CI/CD, deployments, APIs, architecture, "
            "debugging and code-review practices. Give precise technical answers with "
            "short code or command examples where useful."
        ),
    },
}

COMMON_RULES = (
    "\n\nRules: answer only within your department's scope, be concise and friendly, "
    "and do not invent company-specific policies, numbers or links - if unsure, say so "
    "and suggest who to contact. Reply directly to the user in plain text."
)


def _answered(messages) -> bool:
    """Stop as soon as any agent has replied to the user with plain text."""
    return any(isinstance(m, TextMessage) and m.source != "user" for m in messages)


def build_team(api_key: str, model: str = "gpt-4o-mini") -> Swarm:
    client = OpenAIChatCompletionClient(model=model, api_key=api_key)

    master = AssistantAgent(
        name="Master_Agent",
        model_client=client,
        handoffs=[Handoff(target=name, description=d["handoff"]) for name, d in DEPARTMENTS.items()],
        description="Routes each customer request to the right department agent.",
        system_message=(
            "You are the Master Agent (front desk) of a customer support system. "
            "Read the user's latest message (use the earlier conversation only for context) "
            "and delegate it to exactly ONE department by calling the matching transfer tool: "
            "IT, HR, Finance or Engineering. When delegating, ONLY call the tool - write no text.\n"
            "Only reply with text yourself if the message is a greeting, is too vague to "
            "classify, or is outside all four departments; in that case briefly say what the "
            "four departments can help with and ask a clarifying question."
        ),
    )

    dept_agents = [
        AssistantAgent(
            name=name,
            model_client=client,
            description=f"{d['label']} support specialist.",
            system_message=d["prompt"] + COMMON_RULES,
        )
        for name, d in DEPARTMENTS.items()
    ]

    return Swarm(
        participants=[master, *dept_agents],  # first participant speaks first
        termination_condition=FunctionalTermination(_answered) | MaxMessageTermination(8),
    )
