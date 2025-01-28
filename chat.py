from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

conversation_history = [
    {"role": "system", "content": "You are a helpful assistant"}
]

def chat_with_assistant(userInput):
    # Add the user's message to the conversation history
    conversation_history.append({"role": "user", "content": userInput})

    # Get the assistant's response based on the updated conversation history
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=conversation_history,
        stream=False
    )

    assistant_reply = response.choices[0].message.content

    conversation_history.append({"role": "assistant", "content": assistant_reply})

    return assistant_reply

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    assistant_reply = chat_with_assistant(user_input)
    print("Assistant:", assistant_reply)
