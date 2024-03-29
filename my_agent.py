import pandas as pd
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
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