# feight_rag
一个Agentic RAG 的货代智能专家系统
用户首先启动PostgreSQL数据库，然后依据货代私有数据是否更新的情况。然后启动服务器的大模型服务，连接本地大模型或使用线上大模型的API。如果货代数据新增则运行vectorSave.py文件，重新生成向量数据库。如果没有新增数据，运行vectorSave.py文件也可以正常使用，只是会浪费算力。测试ragAgent.py，无误后streamlit run chat.py。浏览器访问http://localhost:8501，用户即可进入货代专家系统聊天界面。

<img width="253" height="404" alt="image" src="https://github.com/user-attachments/assets/1cc0a73d-668e-414a-99e6-58baf7d82729" />
