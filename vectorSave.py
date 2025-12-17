# 功能说明：将PDF文件进行向量计算并持久化存储到向量数据库（chroma）
import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import logging
from openai import OpenAI
import chromadb
import uuid
from utils import pdfSplitTest_Ch
from utils import pdfSplitTest_En
from pathlib import Path
# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# GPT大模型 OpenAI相关配置
OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# 国产大模型 OneAPI相关配置,通义千问为例
ONEAPI_API_BASE = "http://139.224.72.218:3000/v1"
ONEAPI_EMBEDDING_API_KEY = "*******"
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"

# 阿里通义千问大模型
QWen_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWen_EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWen_EMBEDDING_MODEL = "text-embedding-v4"

# 本地开源大模型 vLLM 方式
# 本地开源大模型 xinference 方式
# 本地开源大模型 Ollama 方式,bge-m3为例
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_EMBEDDING_API_KEY = "ollama"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"


# openai:调用gpt模型, qwen:调用阿里通义千问大模型, oneapi:调用oneapi方案支持的模型, ollama:调用本地开源大模型
llmType = "ollama"

# 设置测试文本类型 Chinese 或 English
TEXT_LANGUAGE = 'Chinese'
INPUT_FOLDER = "./pdf_files"
# TEXT_LANGUAGE = 'English'
# INPUT_PDF = "input/deepseek-v3-1-4.pdf"

def get_all_pdf_files(folder_path):
    """获取文件夹下所有PDF文件的绝对路径"""
    pdf_files = []
    # 遍历文件夹
    for file in Path(folder_path).glob("*.pdf"):  # 只处理一级目录，不递归子文件夹
        if file.suffix.lower() == ".pdf":
            pdf_files.append(str(file.absolute()))
    if not pdf_files:
        logger.warning(f"在文件夹 {folder_path} 中未找到任何PDF文件")
    else:
        logger.info(f"找到 {len(pdf_files)} 个PDF文件待处理")
    return pdf_files

# 指定文件中待处理的页码，全部页码则填None
PAGE_NUMBERS=None
# PAGE_NUMBERS=[2, 3]

# 指定向量数据库chromaDB的存储位置和集合 根据自己的实际情况进行调整
CHROMADB_DIRECTORY = "chromaDB"  # chromaDB向量数据库的持久化路径
CHROMADB_COLLECTION_NAME = "demo001"  # 待查询的chromaDB向量数据库的集合名称


# get_embeddings方法计算向量
def get_embeddings(texts):
    global llmType
    global ONEAPI_API_BASE, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    global QWen_API_BASE, QWen_EMBEDDING_API_KEY, QWen_EMBEDDING_MODEL
    global OLLAMA_API_BASE, OLLAMA_EMBEDDING_API_KEY, OLLAMA_EMBEDDING_MODEL
    if llmType == 'oneapi':
        try:
            client = OpenAI(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=ONEAPI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'qwen':
        try:
            client = OpenAI(
                base_url=QWen_API_BASE,
                api_key=QWen_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=QWen_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'ollama':
        try:
            client = OpenAI(
                base_url=OLLAMA_API_BASE,
                api_key=OLLAMA_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OLLAMA_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    else:
        try:
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []


# 对文本按批次进行向量计算
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # 调用向量生成get_embeddings方法  根据调用的API不同进行选择
        response = get_embeddings(batch)
        results.extend(response)
    return results


# 封装向量数据库chromadb类，提供两种方法
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        # 申明使用全局变量
        global CHROMADB_DIRECTORY
        # 实例化一个chromadb对象
        # 设置一个文件夹进行向量数据库的持久化存储  路径为当前文件夹下chromaDB文件夹
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        # 创建一个collection数据集合
        # get_or_create_collection()获取一个现有的向量集合，如果该集合不存在，则创建一个新的集合
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        # embedding处理函数
        self.embedding_fn = embedding_fn

    # 添加文档到集合
    # 文档通常包括文本数据和其对应的向量表示，这些向量可以用于后续的搜索和相似度计算
    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 调用函数计算出文档中文本数据对应的向量
            documents=documents,  # 文档的文本数据
            ids=[str(uuid.uuid4()) for i in range(len(documents))]  # 文档的唯一标识符 自动生成uuid,128位  
        )
        
    # 检索向量数据库，返回包含查询结果的对象或列表，这些结果包括最相似的向量及其相关信息
    # query：查询文本
    # top_n：返回与查询向量最相似的前 n 个向量
    def search(self, query, top_n):
        try:
            results = self.collection.query(
                # 计算查询文本的向量，然后将查询文本生成的向量在向量数据库中进行相似度检索
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.info(f"检索向量数据库时出错: {e}")
            return []


# 封装文本预处理及灌库方法, 提供外部调用
# def vectorStoreSave():
#     global TEXT_LANGUAGE, CHROMADB_COLLECTION_NAME, INPUT_PDF, PAGE_NUMBERS
#
#     # 测试中文文本
#     if TEXT_LANGUAGE == 'Chinese':
#         # 1、获取处理后的文本数据
#         # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
#         paragraphs = pdfSplitTest_Ch.getParagraphs(
#             filename=INPUT_PDF,
#             page_numbers=PAGE_NUMBERS,
#             min_line_length=1
#         )
#         # 2、将文本片段灌入向量数据库
#         # 实例化一个向量数据库对象
#         # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
#         vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
#         # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
#         vector_db.add_documents(paragraphs)
#         # 3、封装检索接口进行检索测试
#         user_query = "张三九的基本信息是什么"
#         # 将检索出的5个近似的结果
#         search_results = vector_db.search(user_query, 5)
#         logger.info(f"检索向量数据库的结果: {search_results}")
#
#     # 测试英文文本
#     elif TEXT_LANGUAGE == 'English':
#         # 1、获取处理后的文本数据
#         # 演示测试对指定的全部页进行处理，其返回值为划分为段落的文本列表
#         paragraphs = pdfSplitTest_En.getParagraphs(
#             filename=INPUT_PDF,
#             page_numbers=PAGE_NUMBERS,
#             min_line_length=1
#         )
#         # 2、将文本片段灌入向量数据库
#         # 实例化一个向量数据库对象
#         # 其中，传参collection_name为集合名称, embedding_fn为向量处理函数
#         vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
#         # 向向量数据库中添加文档（文本数据、文本数据对应的向量数据）
#         vector_db.add_documents(paragraphs)
#         # 3、封装检索接口进行检索测试
#         user_query = "deepseek V3有多少参数"
#         # 将检索出的5个近似的结果
#         search_results = vector_db.search(user_query, 5)
#         logger.info(f"检索向量数据库的结果: {search_results}")


def vectorStoreSave():
    global TEXT_LANGUAGE, CHROMADB_COLLECTION_NAME, INPUT_FOLDER  # 改为文件夹配置

    # 获取所有PDF文件路径
    pdf_files = get_all_pdf_files(INPUT_FOLDER)
    if not pdf_files:
        return  # 无文件则退出

    # 初始化向量数据库连接
    vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)

    # 累计所有PDF的文本片段
    all_paragraphs = []

    # 遍历处理每个PDF文件
    for pdf_file in pdf_files:
        logger.info(f"开始处理文件: {pdf_file}")
        try:
            # 根据语言选择对应的处理函数
            if TEXT_LANGUAGE == 'Chinese':
                paragraphs = pdfSplitTest_Ch.getParagraphs(
                    filename=pdf_file,
                    page_numbers=None,  # 处理所有页
                    min_line_length=1
                )
            elif TEXT_LANGUAGE == 'English':
                paragraphs = pdfSplitTest_En.getParagraphs(
                    filename=pdf_file,
                    page_numbers=None,
                    min_line_length=1
                )
            else:
                logger.error(f"不支持的语言类型: {TEXT_LANGUAGE}")
                continue

            # 累计文本片段
            all_paragraphs.extend(paragraphs)
            logger.info(f"文件 {pdf_file} 处理完成，提取到 {len(paragraphs)} 个文本片段")

        except Exception as e:
            logger.error(f"处理文件 {pdf_file} 时出错: {e}", exc_info=True)
            continue  # 跳过错误文件，继续处理下一个

    # 批量添加所有文本片段到向量数据库
    if all_paragraphs:
        logger.info(f"开始向向量数据库添加 {len(all_paragraphs)} 个文本片段...")
        vector_db.add_documents(all_paragraphs)
        logger.info("所有PDF文件的文本片段已成功添加到向量数据库")

        # 测试检索（可选）
        test_query = "货代的主要工作是？" if TEXT_LANGUAGE == 'Chinese' else "deepseek V3 parameters"
        search_results = vector_db.search(test_query, 5)
        logger.info(f"检索测试结果: {search_results}")
    else:
        logger.warning("未提取到任何文本片段，向量数据库未更新")

if __name__ == "__main__":
    # 测试文本预处理及灌库
    vectorStoreSave()

