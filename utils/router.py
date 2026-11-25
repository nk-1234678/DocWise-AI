
def is_document_question(client, user_text, messages):
    """
    Decide whether the user's latest question
    is related to an uploaded PDF or general chat.

    Returns:
        True  -> PDF/document-related question
        False -> General conversation
    """

    history = []

    # Include recent conversation for context
    for message in messages[-8:]:
        role = "User" if message["role"] == "user" else "Assistant"

        history.append(
            f"{role}: {message['content']}"
        )

    conversation = "\n".join(history)

    prompt = f"""
You are the question router for DocWise AI.

Your task is to classify the user's latest message.

Return ONLY one word:
DOCUMENT
or
GENERAL

Choose DOCUMENT when:
- The user asks about an uploaded PDF.
- The user asks to summarize or explain their document.
- The user refers to their file, PDF, document, or its content.
- The user asks a follow-up question about information
  previously discussed from the uploaded PDF.

Choose GENERAL when:
- The user says hello, hi, or asks casual questions.
- The user asks general knowledge questions.
- The user asks about programming, technology, or other
  topics without referring to their uploaded document.
- The question is unrelated to uploaded PDF content.

Previous conversation:
{conversation}

Latest user message:
{user_text}

Classification:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    decision = (response.text or "").strip().upper()

    return decision.startswith("DOCUMENT")
