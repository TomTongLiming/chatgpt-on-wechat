from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage

chat_model = ChatOllama(
    model="llama2",
)

messages = [HumanMessage(content="hi")]
# messages = [HumanMessage(content="我买的自行车左侧脚踏安装不上，请帮我换一个新的")]
response = chat_model(messages)
print(response)