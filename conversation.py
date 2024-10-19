# conversation.py

conversation_history = []

def update_conversation(user_input, bot_response):
    conversation_history.append({'user': user_input, 'bot': bot_response})
    # Limit history to last 5 turns
    if len(conversation_history) > 5:
        conversation_history.pop(0)

def get_conversation_context():
    context = ""
    for turn in conversation_history:
        context += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    return context
