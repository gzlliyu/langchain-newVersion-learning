# 引入uuid库，用于生成唯一标识符
import uuid  
from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建一个TextLoader对象，用于加载文本文件
loader = TextLoader("xx.txt")
# 创建一个RecursiveCharacterTextSplitter对象，用于分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)  
# 使用TextLoader对象加载文本文件
docs = loader.load()  
# 使用RecursiveCharacterTextSplitter对象分割文档
docs = text_splitter.split_documents(docs)  

# 定义一个字典，键为"doc"，值为一个函数，该函数返回文档的内容
chain = (
    {"doc": lambda x: x.page_content}  
    # 使用ChatPromptTemplate类的from_template方法创建一个聊天提示模板
    | ChatPromptTemplate.from_template("总结以下文档:\n\n{doc}")  
    # 创建一个ChatOpenAI对象，最大重试次数为0
    | ChatOpenAI(max_retries=0)  
    # 创建一个StrOutputParser对象
    | StrOutputParser()  
)
# 使用chain对象的batch方法批量处理文档，最大并发数为5
summaries = chain.batch(docs, {"max_concurrency": 5})  

# 创建一个Chroma对象，集合名为"summaries"，嵌入函数为OpenAIEmbeddings()
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())  
# 创建一个InMemoryByteStore对象，用于存储字节数据
store = InMemoryByteStore()  
# 定义一个字符串，值为"doc_id"
id_key = "doc_id"  
# 创建一个MultiVectorRetriever对象
retriever = MultiVectorRetriever(  
    # 设置向量存储为vectorstore对象
    vectorstore=vectorstore,  
    # 设置字节存储为store对象
    byte_store=store,  
    # 设置id键为"id_key"
    id_key=id_key,  
)
# 为每个文档生成一个唯一的ID
doc_ids = [str(uuid.uuid4()) for _ in docs]  
# 创建一个Document对象，页面内容为s，元数据为{id_key: doc_ids[i]}
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})  
    # 对summaries中的每一个元素和它的索引进行迭代
    for i, s in enumerate(summaries)  
]
# 使用retriever对象的vectorstore属性的add_documents方法添加文档
retriever.vectorstore.add_documents(summary_docs)  
# 使用retriever对象的docstore属性的mset方法设置文档
retriever.docstore.mset(list(zip(doc_ids, docs)))  
# 使用vectorstore对象的similarity_search方法搜索与"如何写文章"相似的文档
sub_docs = vectorstore.similarity_search("如何写文章")  
# 获取搜索结果的第一个文档
sub_docs[0]  
# 使用retriever对象的get_relevant_documents方法获取与"如何写文章"相关的文档
retrieved_docs = retriever.get_relevant_documents("如何写文章")  
# 获取搜索结果的第一个文档的内容
retrieved_docs[0].page_content  
