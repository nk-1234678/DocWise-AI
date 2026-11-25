
def generate_general_answer(client, messages, user_text):
    """
    Generate a streaming response for normal conversations.
    """

    history = []

    for message in messages[-12:]:
        role = "user" if message["role"] == "user" else "model"

        history.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })

    history.append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    response = client.models.generate_content_stream(
        model="gemini-3.5-flash",
        contents=history,
        config={
            "system_instruction": """
            You are DocWise AI, a friendly and helpful AI assistant.

            - Have natural conversations with the user.
            - Answer general questions using your knowledge.
            - Remember previous conversation context.
            - Give clear and easy-to-understand explanations.
            - Use examples whenever helpful.
            - If you are unsure, be honest.
            - Do not assume every question is related to an uploaded PDF.
            """
        }
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text
