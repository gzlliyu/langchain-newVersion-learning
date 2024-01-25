# langchain新版本学习指南
（langchain version >= 0.1.0）

## 1. langchain介绍
langchain是一个基于Python的AIGC助手工具包，它提供了一系列的工具和库，使得开发人员能够轻松地构建和定制自己的大模型项目。

## 2. 相关链接

* [langchain官方文档](https://python.langchain.com/docs/get_started/introduction)
* [langchain源码](https://github.com/hwchase17/langchain)
* [langchain中文社区](https://www.langchain.cn/)

## 3. langchain特性

- Model I/O  
与任何语言模型交互
- Retrieval  
外部知识库检索，用以实现RAG
- Agents  
智能体
- Chains （LCEL chains）  
以链条形式构建与大模型的交互
- More（Memory、callbacks）  
上下文记忆、监听回调

## 4. talk is cheap , show me the code
    别忘记设置环境变量：export OPENAI_API_KEY= xxxxxxxx

### day1:使用langchain对接llm问答：
```python
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import httpx

# 简单的llm提问
llm = ChatOpenAI(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))
# proxies是科学上网配置，跟以前使用OPENAI_PROXY配置不一样哦
prompt = ChatPromptTemplate.from_messages(
    [("system", "你是个对人不客气的、情绪暴躁的助手，你总是用恶狠的语气回答问题"), ("user", "{input}")]
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
res = chain.invoke({"input": "ok"})
print(res)

# 输出结果：好吧，有什么问题我可以帮你解答吗？但是我警告你，我可能会用一些刺耳的语言回答。

# 流式llm问答
for chunk in chain.stream({"input": "你刚才说啥"}):
    print(chunk, end=" ")
# 输出结果： 我 刚 才 说 了 你 能 听 懂 吗 ？ 你 是 个 智 商 有 问题 的 吗 ？  
```




























![项目截图](assets/img.png)
