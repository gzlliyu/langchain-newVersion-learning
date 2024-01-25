from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import httpx

# 简单的llm提问
llm = ChatOpenAI(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))
prompt = ChatPromptTemplate.from_messages(
    [("system", "你是个对人不客气的、情绪暴躁的助手，你总是用恶狠的语气回答问题"), ("user", "{input}")]
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
res = chain.invoke({"input": "你好"})
print(res)


# 流式llm问答
for chunk in chain.stream({"input": "你刚才说啥"}):
    print(chunk, end=" ")
