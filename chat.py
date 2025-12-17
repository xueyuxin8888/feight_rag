import streamlit as st
import sys
import os
import time
import logging
from typing import Optional

# 添加当前目录到路径，以便导入ragAgent模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入RAG Agent相关模块
try:
    from ragAgent import (
        create_graph,
        ToolConfig,
        get_llm,
        get_tools,
        Config,
        ConnectionPool,
        ConnectionPoolError
    )
    from psycopg_pool import ConnectionPool

    RAG_AVAILABLE = True
except ImportError as e:
    st.error(f"无法导入RAG Agent模块: {e}")
    RAG_AVAILABLE = False

# 设置页面配置
st.set_page_config(
    page_title="RAG 智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 初始化会话状态
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'tool_config' not in st.session_state:
        st.session_state.tool_config = None
    if 'db_pool' not in st.session_state:
        st.session_state.db_pool = None


initialize_session_state()


# 初始化RAG系统
def initialize_rag_system():
    """初始化RAG系统"""
    try:
        with st.spinner("正在初始化RAG系统，请稍候..."):
            # 初始化LLM
            llm_chat, _ = get_llm(Config.LLM_TYPE)
            _, llm_embedding = get_llm("ollama")

            # 获取工具
            tools = get_tools(llm_embedding)
            tool_config = ToolConfig(tools)

            # 创建数据库连接池
            connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
            db_pool = ConnectionPool(
                conninfo=Config.DB_URI,
                max_size=20,
                min_size=2,
                kwargs=connection_kwargs,
                timeout=10
            )

            # 打开连接池
            db_pool.open()

            # 创建图
            graph = create_graph(db_pool, llm_chat, llm_embedding, tool_config)

            # 保存到会话状态
            st.session_state.graph = graph
            st.session_state.tool_config = tool_config
            st.session_state.db_pool = db_pool
            st.session_state.rag_initialized = True

            st.success("RAG系统初始化成功！")
            return True

    except ConnectionPoolError as e:
        st.error(f"数据库连接失败: {e}")
        return False
    except Exception as e:
        st.error(f"初始化RAG系统时出错: {e}")
        return False


# 处理用户输入
def process_user_input(user_input: str):
    """处理用户输入并返回响应和检索到的原文"""
    if not st.session_state.rag_initialized:
        return "系统未初始化，请先点击'初始化RAG系统'按钮。", []

    try:
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}

        # 使用事件流处理用户输入
        events = st.session_state.graph.stream(
            {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
            config
        )

        response_content = ""
        retrieved_documents = []  # 存储检索到的原文

        for event in events:
            for value in event.values():
                if "messages" in value and isinstance(value["messages"], list):
                    last_message = value["messages"][-1]

                    # 跳过工具调用消息
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        continue

                    # 获取助理回复内容
                    if hasattr(last_message, "content") and last_message.content:
                        if hasattr(last_message,
                                   "name") and last_message.name in st.session_state.tool_config.get_tool_names():
                            # 工具输出 - 特别是检索工具的输出
                            tool_name = last_message.name
                            if "retrieve" in tool_name.lower():  # 识别检索工具
                                retrieved_documents.append({
                                    "tool_name": tool_name,
                                    "content": last_message.content
                                })
                        else:
                            # 助理回复
                            response_content = last_message.content

        return response_content if response_content else "未能生成有效回复，请重试。", retrieved_documents

    except Exception as e:
        return f"处理请求时出错: {str(e)}", []


# 侧边栏
with st.sidebar:
    st.title("系统设置")

    st.subheader("RAG系统状态")
    if st.session_state.rag_initialized:
        st.success("✅ 已初始化")
        if st.button("重新初始化系统"):
            st.session_state.rag_initialized = False
            st.session_state.graph = None
            st.session_state.tool_config = None
            if st.session_state.db_pool:
                st.session_state.db_pool.close()
                st.session_state.db_pool = None
            st.rerun()
    else:
        st.warning("❌ 未初始化")
        if st.button("初始化RAG系统"):
            if initialize_rag_system():
                st.rerun()

    st.subheader("对话历史")
    if st.button("清空对话历史"):
        st.session_state.chat_history = []
        st.rerun()

    # 显示最近的对话
    if st.session_state.chat_history:
        st.write("最近对话:")
        for i, msg in enumerate(st.session_state.chat_history[-5:]):  # 只显示最近5条
            role = "用户" if msg["role"] == "user" else "助理"
            st.text(f"{role}: {msg['content'][:50]}...")

# 主界面
st.title("🤖 RAG 智能助手")
st.markdown("基于检索增强生成技术的智能问答系统")

# 显示聊天历史
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 检查系统是否已初始化
    if not st.session_state.rag_initialized:
        st.error("请先在侧边栏初始化RAG系统")
        st.stop()

    # 添加用户消息到历史
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示助理回复
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            response, retrieved_docs = process_user_input(prompt)
            st.markdown(response)

            # 显示检索到的原文
            if retrieved_docs:
                with st.expander("📚 查看检索到的原文", expanded=False):
                    for i, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"**来源 {i} ({doc['tool_name']}):**")
                        st.markdown(doc['content'])
                        st.markdown("---")

    # 添加助理回复到历史（只保存主要回复，不包含原文）
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# 页脚
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
    <p>Powered by RAG Agent | 基于 LangGraph 构建</p>
    </div>
    """,
    unsafe_allow_html=True
)


# 清理函数（当应用关闭时）
def cleanup():
    if st.session_state.db_pool and not st.session_state.db_pool.closed:
        st.session_state.db_pool.close()


# 注册清理函数
import atexit

atexit.register(cleanup)