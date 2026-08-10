from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

chatprompttemplate = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    (MessagesPlaceholder(variable_name="chat_history")),
    ("human", "{input}")
])

chat_history_content = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break;
    prompt = chatprompttemplate.invoke({"input": user_input, "chat_history": chat_history_content})
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731:novita",
        messages=prompt
    )

    result = completion.choices[0].message.content
    print("Assistant:", result)

    chat_history_content.append(HumanMessage(content=user_input))
    chat_history_content.append(AIMessage(content=result))

