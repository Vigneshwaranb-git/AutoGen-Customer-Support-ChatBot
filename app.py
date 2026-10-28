import asyncio
import os

import streamlit as st
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import HandoffMessage, TextMessage
from dotenv import load_dotenv

from agents import DEPARTMENTS, build_team

load_dotenv()  # reads OPENAI_API_KEY from the .env file
api_key = os.getenv("OPENAI_API_KEY", "")
model = "gpt-4o-mini"

st.set_page_config(page_title="Customer Support Bot", page_icon="🎧", layout="centered")
st.title("🎧 Customer Support Chatbot")
st.caption("Master Agent → IT · HR · Finance · Engineering (built with AutoGen)")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

with st.sidebar:
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("**Try asking**")
    st.markdown(
        "- My VPN keeps disconnecting\n"
        "- How many casual leaves do I get?\n"
        "- How do I claim travel expenses?\n"
        "- How do I roll back a failed deployment?"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "agent"}


def badge(agent: str | None) -> str:
    if agent in DEPARTMENTS:
        d = DEPARTMENTS[agent]
        return f"{d['icon']} {d['label']}"
    return "🎧 Master Agent"


def build_task(history, latest: str) -> str:
    convo = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-8:])
    return f"Earlier conversation:\n{convo or '(none)'}\n\nLatest user message: {latest}"


async def run_turn(task: str, route_box):
    team = build_team(api_key, model)  # fresh team each turn -> always starts at Master_Agent
    agent, answer = None, None
    async for msg in team.run_stream(task=task):
        if isinstance(msg, TaskResult):
            break
        if isinstance(msg, HandoffMessage) and msg.source == "Master_Agent":
            route_box.info(f"🔀 Master Agent delegated to **{badge(msg.target)}**")
        elif isinstance(msg, TextMessage) and msg.source != "user":
            agent, answer = msg.source, msg.content
    return agent, answer


# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            st.caption(badge(m["agent"]))
        st.markdown(m["content"])

prompt = st.chat_input("How can we help you today?")

if prompt:
    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt, "agent": None})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        route_box = st.empty()
        with st.spinner("Thinking..."):
            try:
                agent, answer = asyncio.run(run_turn(build_task(history, prompt), route_box))
            except Exception as e:
                agent, answer = None, f"Sorry, something went wrong: {e}"
        answer = answer or "Sorry, I couldn't produce a reply. Please try rephrasing."
        st.caption(badge(agent))
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "agent": agent})
