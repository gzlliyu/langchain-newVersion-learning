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


