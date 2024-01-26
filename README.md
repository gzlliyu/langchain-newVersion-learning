# langchain新版本学习指南
（langchain version >= 0.1.0）

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
 Lang Chain 是 一个 开 发 应 用 程序 的 框 架 ， 它 利 用 语 言 模 型 的 能 力 。 它 可以 创建 具 有 以下 特 点 的 应 用 程序 ：
 -  上 下 文 感 知 ： 将 语 言 模 型 连接 到 上 下 文 的 信息 源 （ 提示 指 令 、 少 量 示 例 、 内容 等 ）， 使 其 响 应 基 于 提 供 的 上 下 文 进行 推 理 。
 -  推 理 ： 依 赖 语 言 模 型 进行 推 理 ， 例如 根 据 提 供 的 上 下 文 选择 如 何 回 答 和 采 取 哪 些 操作 等 。

 Lang Chain 框 架 由 几 个 部 分 组 成 ：
 -  Lang Chain 库 ： Python 和 JavaScript 库 ， 包 含 用 于 各 种 组 件 的 接 口 和 集 成 、 将 这 些 组 件 组 合 成 链 和 代 理 的 基 本 运 行 时 ， 以 及 一 些 现 成 的 链 和 代 理 实 现 。
 -  Lang Chain 模 板 ： 一 系 列 易 于 部 署 的 参 考 架 构 ， 适 用 于 各 种 任务 。
 -  Lang Serve ： 将 Lang Chain 链 部 署 为 REST  API 的 库 。
 -  Lang Smith ： 开 发 平 台 ， 可 用 于 调 试 、 测试 、 评 估 和 监 控 基 于 任 何 LL M 框 架 构 建 的 链 ，并 与 Lang Chain 无 缝 集 成 。

 这 些 产品 共 同 简 化 了 整 个 应 用 程序 生 命周期 ：
 -  开 发 ： 使用 Lang Chain /L ang Chain .js 编 写 应 用 程序 ， 使用 模 板 作 为 参 考 ， 快 速 上 手 。
 -  生 产 ： 使用 Lang Smith 检 查 、 测试 和 监 控 您 的 链 ， 以 便 不 断 改 进 并 自 信 地 部 署 。
 -  部 署 ： 使用 Lang Serve 将 任 何 链 转 换 为 API 。

 Lang Chain 库 的 主 要 价 值 在 于 ：
 -  组件 ： 可 组 合 的 工 具 和 语 言 模 型 集 成 。 组 件 是 模 块 化 且 易 于 使用 的 ， 无 论 您 是否 使用 Lang Chain 框 架 的 其他 部 分 。
 -  现 成 的 链 ： 内 置 的 组 件 集 合 ， 用 于 完成 更 高 级 的 任务 。 现 成 的 链 使 得 快 速 入 门 变 得 容 易 ， 组 件 使 得 定 制 现 有 链 和 构 建 新 链 变 得 容 易 。

 Lang Chain 提 供 了 标 准 、 可 扩 展 的 接 口 和 集 成 ， 用 于 以下 模 块 ：
 -  模 型 I /O ： 与 语 言 模 型 进行 交 互 。
 -  检 索 ： 与 特 定 应 用 程序 数据 进行 交 互 。
 -   代 理 ： 让 模 型 根 据 高 级 指 令 选择 使用 哪 些 工 具 。

 Lang Chain 还 与 丰 富 的 工 具 生 态 系统 进行 集 成 ， 可以 构 建 在 其 之 上 。 此 外 ， Lang Chain 还 提 供 了 使用 指 南 、 API 参 考 和 开 发 者 指 南 等 资源 供 开 发 者 参 考 。  
```