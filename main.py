# swarm_backend.py
"""
Dedalus Labs swarm-agent backend (single-file).
- Async implementation using AsyncDedalus + DedalusRunner
- Classifier agent routes user input to sub-agents
- Each sub-agent calls the model to produce a structured list of constraints & micro-solutions
- Synchroniser aggregates the stack and resolves conflicts
Notes:
- Replace model names / MCP servers with your desired values.
- This is designed for hackathon-quality prototyping and is intentionally modular.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable

from dotenv import load_dotenv
from dedalus_labs import AsyncDedalus, DedalusRunner

load_dotenv()

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("swarm-backend")

# -------------------------
# Data structures
# -------------------------
@dataclass
class AgentResult:
    agent_name: str
    constraints: List[str]
    micro_solutions: List[str]
    metadata: Dict[str, Any]

    def to_dict(self):
        return asdict(self)


class SharedStack:
    """A thread-safe shared stack (list) for agent outputs."""
    def __init__(self):
        self._stack: List[AgentResult] = []
        self._lock = asyncio.Lock()

    async def push(self, item: AgentResult):
        async with self._lock:
            self._stack.append(item)
            logger.info(f"Pushed result from {item.agent_name} (stack size={len(self._stack)})")

    async def get_all(self) -> List[AgentResult]:
        async with self._lock:
            # return a shallow copy so aggregator can operate safely
            return list(self._stack)

    async def clear(self):
        async with self._lock:
            self._stack.clear()
            logger.info("Stack cleared.")

# -------------------------
# Utility: model call wrapper
# -------------------------
async def call_model(
    runner: DedalusRunner,
    prompt: str,
    *,
    model: Any = "openai/gpt-5",
    mcp_servers: Optional[List[str]] = None,
    stream: bool = False,
    timeout: int = 30
) -> str:
    """
    Call Dedalus Runner to query the model. Returns plain text final output.
    Keep runs short in hackathon mode by using focused prompts.
    """
    try:
        # The exact call format depends on DedalusRunner API; previously you used runner.run(...)
        result = await runner.run(
            input=prompt,
            model=model,
            mcp_servers=mcp_servers or [],
            stream=stream
        )
        # result.final_output is used in prior examples
        return getattr(result, "final_output", str(result))
    except Exception as e:
        logger.exception("Model call failed; returning fallback.")
        return f"ERROR: model call failed: {e}"


# -------------------------
# Prompt templates
# -------------------------
CLASSIFIER_PROMPT = """
You are the CLASSIFIER AGENT. A user has provided the following prompt:
"{user_input}"

Your job:
1. Determine which subagents are relevant from this list:
   [Technical & Engineering, Design & Performance, Team & Management, Business & Resources, Legal & Ethical, Deployment & Feedback]
2. Return a JSON object with:
   - "selected_agents": list of agent short names (see mapping below)
   - "notes": short explanation (1-2 sentences) of why each agent was selected

Return only valid JSON (no extra text).

Mapping of short names:
  technical -> Technical & Engineering
  design -> Design & Performance
  team -> Team & Management
  business -> Business & Resources
  legal -> Legal & Ethical
  deploy -> Deployment & Feedback
"""

SUBAGENT_PROMPT_TEMPLATE = """
You are the {agent_title} sub-agent. You were given this user input:
"{user_input}"

Guidelines:
- Produce a JSON object with keys:
  - "constraints": list of short constraints (1-12 items). Each constraint must be 6-30 words.
  - "micro_solutions": list of short actionable micro-solutions mapping to constraints (same length).
  - "rationale": 1-2 sentence explanation of how you formed the constraints.
- If you need clarifying details from the user to produce higher fidelity constraints, set "need_more_info": true and optionally include "questions": [ ... ].
- Otherwise set "need_more_info": false.

Return only JSON.
"""

SYNCHRONISER_PROMPT_TEMPLATE = """
You are the SYNCHRONISER. You will receive a JSON array of agent outputs (constraints + micro_solutions).
Your job:
- Detect duplicate constraints or direct conflicts and produce a reconciled plan.
- Mark conflicts and propose a preferred choice with reasoning.
- Output a final consolidated JSON object:
  - "final_constraints": list of constraints
  - "final_solutions": list of micro solutions
  - "conflicts": list of { "between": [agent1, agent2], "issue": "...", "resolution": "..." }
  - "notes": short summary (2-4 sentences)
Return only JSON.
"""

# -------------------------
# Agents
# -------------------------
async def classifier_agent(
    user_input: str,
    runner: DedalusRunner,
    *,
    model: Any = "openai/gpt-5-mini",
    mcp_servers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Classify user input to relevant sub-agents using the model (or a heuristic fallback)."""
    prompt = CLASSIFIER_PROMPT.format(user_input=user_input)
    raw = await call_model(runner, prompt, model=model, mcp_servers=mcp_servers)
    # Model expected to return JSON. Attempt parse; fallback to heuristics.
    try:
        parsed = json.loads(raw)
        logger.info(f"Classifier parsed JSON: {parsed}")
        return parsed
    except Exception:
        # Heuristic fallback: keyword matching
        logger.warning("Classifier model output not JSON; using heuristic fallback.")
        low = user_input.lower()
        sel = set()
        if any(k in low for k in ["sensor", "actuator", "pcb", "power", "battery", "controller", "can", "lor a", "wifi"]):
            sel.add("technical")
        if any(k in low for k in ["safety", "certification", "compliance", "fcc", "faa", "iso"]):
            sel.add("legal")
        if any(k in low for k in ["cost", "budget", "vendor", "supply", "ip", "patent"]):
            sel.add("business")
        if any(k in low for k in ["team", "hiring", "roles", "timeline", "milestone", "gantt"]):
            sel.add("team")
        if any(k in low for k in ["test", "field", "deployment", "pilot", "feedback"]):
            sel.add("deploy")
        if any(k in low for k in ["speed", "accuracy", "latency", "durability"]):
            sel.add("design")
        if not sel:
            # if nothing matches, default to technical + design + deploy
            sel.update(["technical", "design", "deploy"])
        return {"selected_agents": list(sel), "notes": "Heuristic selection based on keywords."}


async def sub_agent_worker(
    agent_short_name: str,
    user_input: str,
    runner: DedalusRunner,
    stack: SharedStack,
    *,
    model: Any = "openai/gpt-5",
    mcp_servers: Optional[List[str]] = None,
) -> AgentResult:
    """Generic sub-agent worker that instructs the model to produce constraints + micro-solutions."""
    mapping = {
        "technical": "Technical & Engineering",
        "design": "Design & Performance",
        "team": "Team & Management",
        "business": "Business & Resources",
        "legal": "Legal & Ethical",
        "deploy": "Deployment & Feedback"
    }
    agent_title = mapping.get(agent_short_name, agent_short_name)

    # --- DOMAIN-SPECIFIC PREPROMPT INSERTS ---
    if agent_short_name == "technical":
        user_input += "\nFocus: sensors, controllers, power systems, architecture, and testing."
    elif agent_short_name == "design":
        user_input += "\nFocus: performance metrics, durability, and user experience."
    elif agent_short_name == "team":
        user_input += "\nFocus: team roles, collaboration tools, and project milestones."
    elif agent_short_name == "business":
        user_input += "\nFocus: budget, logistics, and intellectual property."
    elif agent_short_name == "legal":
        user_input += "\nFocus: safety certifications, data privacy, and environmental norms."
    elif agent_short_name == "deploy":
        user_input += "\nFocus: field testing, feedback collection, and iteration cycles."


    prompt = SUBAGENT_PROMPT_TEMPLATE.format(agent_title=agent_title, user_input=user_input)
    logger.info(f"Calling model for sub-agent {agent_title}")
    raw = await call_model(runner, prompt, model=model, mcp_servers=mcp_servers)

    # Try parse JSON response
    try:
        parsed = json.loads(raw)
        constraints = parsed.get("constraints", [])
        micro_solutions = parsed.get("micro_solutions", [])
        metadata = {"rationale": parsed.get("rationale"), "need_more_info": parsed.get("need_more_info", False)}
        result = AgentResult(agent_name=agent_title, constraints=constraints, micro_solutions=micro_solutions, metadata=metadata)
        await stack.push(result)
        return result
    except Exception:
        # If model response is garbage, produce a simple fallback list
        logger.exception(f"Sub-agent {agent_title} returned non-JSON. Using fallback suggestions.")
        fallback_constraints = [f"High-level constraint for {agent_title} #1", f"High-level constraint for {agent_title} #2"]
        fallback_solutions = [f"Micro-solution for {agent_title} #1", f"Micro-solution for {agent_title} #2"]
        metadata = {"rationale": "fallback generated", "need_more_info": False, "raw_output": raw[:400]}
        result = AgentResult(agent_name=agent_title, constraints=fallback_constraints, micro_solutions=fallback_solutions, metadata=metadata)
        await stack.push(result)
        return result


async def orchestrate_subagents(
    selected_agents: List[str],
    user_input: str,
    runner: DedalusRunner,
    stack: SharedStack,
    *,
    model: Any = "openai/gpt-5",
    mcp_servers: Optional[List[str]] = None
) -> List[AgentResult]:
    """Run selected sub-agents concurrently and collect their outputs."""
    workers = [
        sub_agent_worker(agent_short_name=ag, user_input=user_input, runner=runner, stack=stack, model=model, mcp_servers=mcp_servers)
        for ag in selected_agents
    ]
    logger.info(f"Starting {len(workers)} sub-agents concurrently: {selected_agents}")
    results = await asyncio.gather(*workers, return_exceptions=False)
    return results


async def synchroniser_agent(
    runner: DedalusRunner,
    stack: SharedStack,
    *,
    model: Any = "openai/gpt-5",
    mcp_servers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Aggregate the stack using the model (preferred) to reconcile conflicts and produce final plan."""
    items = await stack.get_all()
    logger.info(f"Synchroniser received {len(items)} items.")
    # Build input payload for the synchroniser prompt
    serialized = [item.to_dict() for item in items]
    prompt_payload = json.dumps(serialized, indent=2)
    prompt = SYNCHRONISER_PROMPT_TEMPLATE + "\n\nAgentOutputs:\n" + prompt_payload
    raw = await call_model(runner, prompt, model=model, mcp_servers=mcp_servers)
    try:
        parsed = json.loads(raw)
        return parsed
    except Exception:
        # Basic deterministic aggregator fallback
        logger.warning("Synchroniser model output not JSON; using deterministic fallback aggregator.")
        final_constraints = []
        final_solutions = []
        conflicts = []
        seen = set()
        for item in serialized:
            for c, s in zip(item["constraints"], item["micro_solutions"]):
                if c not in seen:
                    seen.add(c)
                    final_constraints.append(c)
                    final_solutions.append(s)
                else:
                    # note as small conflict (duplicate)
                    conflicts.append({"between": [item["agent_name"]], "issue": "duplicate constraint", "resolution": "deduplicated"})
        notes = "Aggregated constraints with deduplication (fallback)."
        return {
            "final_constraints": final_constraints,
            "final_solutions": final_solutions,
            "conflicts": conflicts,
            "notes": notes
        }

# -------------------------
# Main orchestration
# -------------------------
async def handle_user_request(
    user_input: str,
    runner: DedalusRunner,
    stack: SharedStack,
    *,
    classifier_model: Any = "openai/gpt-5-mini",
    subagent_model: Any = "openai/gpt-5",
    synchroniser_model: Any = "openai/gpt-5",
    mcp_servers: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Full pipeline:
    1. Classify which sub-agents are needed
    2. Orchestrate sub-agents to run concurrently
    3. Run synchroniser to consolidate results
    4. Return the final consolidated plan
    """
    # 1) classifier
    classification = await classifier_agent(user_input, runner, model=classifier_model, mcp_servers=mcp_servers)
    selected_agents = classification.get("selected_agents", [])
    if not selected_agents:
        logger.info("No agents selected by classifier. Defaulting to technical, design, deploy.")
        selected_agents = ["technical", "design", "deploy"]

    # 2) run selected sub-agents
    await orchestrate_subagents(selected_agents, user_input, runner, stack, model=subagent_model, mcp_servers=mcp_servers)

    # 3) synchroniser
    final_plan = await synchroniser_agent(runner, stack, model=synchroniser_model, mcp_servers=mcp_servers)

    return {
        "classification": classification,
        "stack": [s.to_dict() for s in await stack.get_all()],
        "final_plan": final_plan
    }


# -------------------------
# Example main (async)
# -------------------------
async def async_main_example():
    """
    Example entrypoint demonstrating:
    - AsyncDedalus + DedalusRunner
    - Running the whole pipeline end-to-end
    """
    client = AsyncDedalus()
    runner = DedalusRunner(client)

    # Shared stack for the session
    stack = SharedStack()

    # Example user prompt (replace with your input)
    user_input = (
        "We want to build an autonomous RC plane prototype for a university competition. "
        "Focus on sensors, power, flight control, testing, budget of $2,000, and regulatory concerns."
    )

    # MCP servers you want to include (optional)
    mcp_servers = ["windsor/brave-search-mcp"]  # adapt as needed

    logger.info("Handling user request...")
    result = await handle_user_request(
        user_input=user_input,
        runner=runner,
        stack=stack,
        classifier_model="openai/gpt-5-mini",
        subagent_model="openai/gpt-5",
        synchroniser_model="openai/gpt-5",
        mcp_servers=mcp_servers
    )

    # Pretty print results
    print("\n=== CLASSIFICATION ===")
    print(json.dumps(result["classification"], indent=2))

    print("\n=== STACK (agent outputs) ===")
    print(json.dumps(result["stack"], indent=2))

    print("\n=== FINAL PLAN ===")
    print(json.dumps(result["final_plan"], indent=2))


# -------------------------
# Sync wrapper for convenience
# -------------------------
def main():
    """Synchronous wrapper for local testing (calls the async main)."""
    # Removed asyncio.run()
    # asyncio.run(async_main_example())
    pass


if __name__ == "__main__":
    # Call the async function directly
    await async_main_example()