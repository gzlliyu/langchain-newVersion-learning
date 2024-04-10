# 如果有用，期待你的star ~

# langchain新版本学习指南
（langchain version >= 0.1.0）

## 0. 一起学习LangChain
* <img src="https://github.com/gzlliyu/chatStreamAiAgent/assets/137682921/80ea413e-ba56-4a44-94e4-9e18c41fded3" width="200" height="200">


## 1. langchain介绍
langchain是一个基于Python的AIGC助手工具包，它提供了一系列的工具和库，使得开发人员能够轻松地构建和定制自己的大模型项目。

## 2. 相关链接

* [langchain官方文档](https://python.langchain.com/docs/get_started/introduction)
* [langchain源码](https://github.com/hwchase17/langchain)
* [langchain文档RAG问答](https://chat.langchain.com/)
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
from langchain_openai import ChatOpenAI
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
### day2:使用langchain实现RAG
```python
import httpx
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")#文档加载器
docs = loader.load()#解析文档
embeddings = OpenAIEmbeddings(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))#向量方式：OpenAI
splitter = RecursiveCharacterTextSplitter()#文件切割器
doc_list = splitter.split_documents(docs)#切割文件
vector = FAISS.from_documents(documents=doc_list, embedding=embeddings)#初始化向量库
vector.save_local(folder_path='./vectorstore')#向量持久化
loaded_vector = FAISS.load_local(folder_path='./vectorstore', embeddings=embeddings)#从本地加载向量库

llm = ChatOpenAI(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))
prompt = ChatPromptTemplate.from_template('''根据提供的context上下文回答问题:
<context>
{context}
</context>

问题：{input}''')

retrieval = loaded_vector.as_retriever()#检索器
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)#文档链
retrieval_chain = create_retrieval_chain(retriever=retrieval, combine_docs_chain=doc_chain)#检索链

for chunk in retrieval_chain.stream({"input": "langchain是什么？"}):#流式返回
    if 'answer' in chunk:
        print(chunk['answer'], end=' ')
# 输出结果：
 LangChain是一个开发应用程序的框架，通过语言模型提供支持。它具备以下特点：
1. 上下文感知：将语言模型与上下文源（提示指令、少量样本示例、内容等）连接起来。
2. 推理能力：依赖语言模型进行推理（根据提供的上下文回答问题、采取行动等）。
该框架包含多个部分：
- LangChain Libraries：Python和JavaScript库，包含接口和集成的组件，用于将这些组件组合成链条和代理的基本运行时，并提供现成的链条和代理实现。
- LangChain Templates：一系列易于部署的参考架构，适用于各种任务。
- LangServe：用于将LangChain链条部署为REST API的库。
- LangSmith：开发平台，可用于对任何基于LLM框架构建的链条进行调试、测试、评估和监控，并与LangChain无缝集成。
这些产品共同简化了整个应用程序生命周期，包括开发、生产和部署阶段。LangChain库的主要价值在于组件和现成的链条。组件是可组合的工具和集成，无论您是否使用LangChain框架的其他部分，都可以轻松使用。现成的链条使得入门变得简单，而组件则使得可以定制现有链条并构建新的链条。
```

### day3:使用langchain实现agent智能体（my_agent.py）
```python
import pandas as pd
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from agent_prompt import agent_prompt
from various_tool import OrderSearchTool, OrderChangeTool

# 文档向量化 至本地
df = pd.read_excel('./assets/电商客服常见问题回答.xlsx')
res = df.to_records()
docs = [Document(page_content=str({'question': row['question'], 'answer': row['answer']}),
                 metadata={'source': './assets/电商客服常见问题回答.xlsx'}) for index, row in df.iterrows()]
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local('./vectorstore')


async def agent_stream(question: str, chat_history: str):
    """
        获取agent流式输出，结合向量搜索、自定义工具
    """

    # 加载向量库工具
    vs = FAISS.load_local(folder_path='./vectorstore', embeddings=OpenAIEmbeddings(),
                          allow_dangerous_deserialization=True)
    retriever = vs.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name='常见客户问答',
        description="当用户询问商品位置、购买方式、如何退货等问题时使用本工具获取提示信息。如果其他工具都不是你想要的，回答用户问题时必须使用此工具获取提示信息。"
                    "你需要根据用户的最新提问结合部分相关有用的提示回答客户，无关的提示需要忽略"
    )

    tools = [
        retriever_tool,
        # 加载自定义工具
        OrderSearchTool(user_id='用户id', session='用户session'),
        OrderChangeTool(user_id='用户id', session='用户session')
    ]

    # 创建agent
    agent = create_react_agent(ChatOpenAI(), tools, agent_prompt)
    # 创建agent执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 最终返回给用户的流式输出前缀
    final_answer = 'Final Answer:'
    tmp_answer = ''

    async for event in agent_executor.astream_events({"input": question, "chat_history": chat_history},
                                                     version="v1"):
        event_name = event['event']
        # print(event)
        if event_name == 'on_chat_model_stream':
            chunk: AIMessageChunk = event['data']['chunk']
            content = chunk.content
            if content and tmp_answer.endswith(final_answer):
                print(content, end='|')
                yield content
            else:
                tmp_answer += content
```