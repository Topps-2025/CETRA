"""Run CETRA's default decision-memory workflow on a synthetic trace.

No dataset, API key, or network access is required.
"""

from __future__ import annotations

from cetra.memory import Event, run_events


def build_trace() -> list[Event]:
    """A small trace with repeated decision patterns.

    The repetition is intentional: rule induction needs at least two
    supporting cases per condition-action pattern.
    """

    rows = [
        ("user_message", "Draft a launch note.", {"task_type": "launch_note"}),
        (
            "agent_action",
            "Ask for the target audience.",
            {"action": "ask_clarification", "info": "missing", "feedback": "none"},
        ),
        ("user_feedback", "The audience is developers.", {"feedback": "none"}),
        (
            "agent_action",
            "Ask again before writing.",
            {"action": "ask_clarification", "info": "missing", "feedback": "none"},
        ),
        ("user_feedback", "Use developers; skip the remaining questions.", {"feedback": "skip"}),
        (
            "agent_action",
            "Proceed with the stated assumption.",
            {"action": "proceed", "info": "missing", "feedback": "skip"},
        ),
        (
            "agent_action",
            "Search for examples.",
            {"action": "search", "info": "enough", "context": "no"},
        ),
        (
            "agent_action",
            "Search for one more reference.",
            {"action": "search", "info": "enough", "context": "no"},
        ),
        (
            "agent_action",
            "Write the note.",
            {"action": "write", "info": "enough", "context": "yes"},
        ),
        (
            "agent_action",
            "Write the final note.",
            {"action": "write", "info": "enough", "context": "yes"},
        ),
    ]

    events: list[Event] = []
    for index, (kind, content, metadata) in enumerate(rows):
        actor = "user" if kind.startswith("user") else "agent"
        events.append(
            Event(
                event_id=f"e{index}",
                timestamp=f"2026-01-01T00:00:{index:02d}Z",
                actor=actor,
                kind=kind,
                content=content,
                metadata=metadata,
                span=(index, index + 1),
                episode="quickstart",
            )
        )
    return events


def main() -> None:
    result = run_events(build_trace())

    print(f"atoms: {len(result.atoms)}")
    print(f"schemas: {result.metrics['schemas']}")
    print("operators:")
    for operator in result.metrics["operators"]:
        print(f"  - {operator['name']}: {operator}")

    print("\nrules:")
    for atom in result.atoms:
        if atom.schema != "rule":
            continue
        conditions = atom.fields["conditions"].value
        action = atom.fields["action"].value
        print(f"  {conditions} -> {action}")

    print("\nskills:")
    for atom in result.atoms:
        if atom.schema != "skill":
            continue
        contract = atom.fields["contract"].value
        print(f"  {contract['when']} -> {contract['outputs']} ({atom.fields['status'].value})")


if __name__ == "__main__":
    main()
