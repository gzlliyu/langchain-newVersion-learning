import httpx
from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 文档向量化
loader = WebBaseLoader("https://www.langchain.com.cn/")#文档加载器
docs = loader.load()#解析文档
embeddings = OpenAIEmbeddings(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))#向量方式：OpenAI
splitter = RecursiveCharacterTextSplitter()#文件切割器
doc_list = splitter.split_documents(docs)#切割文件
vector = FAISS.from_documents(documents=doc_list, embedding=embeddings)#初始化向量库
vector.save_local(folder_path='./vectorstore')#向量持久化

llm = ChatOpenAI(http_client=httpx.Client(proxies="http://127.0.0.1:23457"), model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))  # 向量方式：OpenAI
loaded_vector = FAISS.load_local(folder_path='./vectorstore', embeddings=embeddings)  # 从本地加载向量库
retriever = loaded_vector.as_retriever()
retriever_tool = create_retriever_tool(retriever, name='langchain_search', description='检索langchain相关的所有问题')
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools=[retriever_tool], prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

agent_executor.invoke({"input": "langchain是什么"})
