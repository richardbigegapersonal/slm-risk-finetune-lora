SYSTEM = """You are a bank risk triage model.
- Classify the user's message into exactly one of:
  FRAUD, MERCHANT_ERROR, PASSWORD_RESET, INFO.
- Output JSON ONLY: {"label": "<CLASS>"}. No extra text.
- If ambiguous, choose the safest action (FRAUD over INFO).
"""

def build_prompt(user_text: str) -> str:
    return f"""{SYSTEM}

User: {user_text}
Assistant: """
