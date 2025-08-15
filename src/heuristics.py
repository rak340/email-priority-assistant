from __future__ import annotations
import re

def summarize_intent(subject: str, body: str, label: str) -> str:
    text = (subject + ". " + body).strip()
    text = re.sub(r"\s+", " ", text)
    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(first_sentence) > 140:
        first_sentence = first_sentence[:137] + "..."
    if label == "Urgent":
        return f"Sender reports a time-sensitive issue: {first_sentence}"
    if label == "Low":
        return f"Informational update with no immediate action: {first_sentence}"
    return f"Standard request or update: {first_sentence}"

def draft_reply(subject: str, body: str, label: str) -> str:
    polite_open = "Hi,"
    if label == "Urgent":
        core = (
            "Thanks for flagging this. We’re looking into it now.\n\n"
            "Next steps:\n"
            "• I’ll run initial checks and update you shortly.\n"
            "• If you have error IDs or logs, please share them.\n\n"
            "Best regards,\nTeam"
        )
    elif label == "Normal":
        core = (
            "Thanks for the details. We’ll proceed and circle back with an update.\n\n"
            "Next steps:\n"
            "• We’ll review and confirm any follow-ups.\n\n"
            "Best regards,\nTeam"
        )
    else:
        core = (
            "Thanks for the update—no action needed right now. We’ll keep this on record.\n\n"
            "Best regards,\nTeam"
        )
    return f"{polite_open}\n\n{core}"
