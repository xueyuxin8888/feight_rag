from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from utils.config import Config
from langchain_tavily import TavilySearch
from langchain_community.tools import Tool



def get_tools(llm_embedding):
    """
    创建并返回工具列表

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储

    Returns:
        list: 工具列表
    """

    # 创建 Chroma 向量存储实例
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是货代知识查询工具，搜索并返回有关货代的知识内容。"
    )

    tavily_search = TavilySearch(max_results=2)

    # 创建TavilySearch工具

    tavily_search_tool = Tool(
        name="tavily_search",
        description="这是网络搜索工具，用于查询最新的网络信息、新闻、事件和公开数据等内容。当需要获取时效性强或不确定的信息时使用。",
        func=tavily_search.run
    )

    # 自定义 multiply 工具
    # @tool
    # def multiply(a: float, b: float) -> float:
    #     """这是计算两个数的乘积的工具，返回最终的计算结果"""
    #     tavily_search = TavilySearch(max_results=2)
    #
    #     return a * b


    # 返回工具列表
    return [retriever_tool, tavily_search_tool]