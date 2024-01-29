import httpx
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', "{input}"),
    ('user', '根据上述对话，生成一个与对话信息相关的搜索查询')
])
chat_history = [
    HumanMessage(content='langchain是什么'),
    AIMessage(content='LangChain是一个开发应用程序的框架，通过语言模型提供支持')
]

embeddings = OpenAIEmbeddings(http_client=httpx.Client(proxies="http://127.0.0.1:23457"))  # 向量方式：OpenAI
loaded_vector = FAISS.load_local(folder_path='./vectorstore', embeddings=embeddings)  # 从本地加载向量库

retriever_chain = create_history_aware_retriever(llm=llm, retriever=loaded_vector.as_retriever(), prompt=prompt)

# res = retriever_chain.invoke({"chat_history": chat_history, 'input': '他是怎么实现的？'})

prompt_f = ChatPromptTemplate.from_messages([
    ('system', '根据下面的上下文回答用户的问题:{context}'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}')
])

document_chain = create_stuff_documents_chain(llm, prompt_f)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

res = retrieval_chain.invoke({
    "chat_history": chat_history,
    'input': "他是怎么实现的"
})

print(res)