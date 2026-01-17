"""
RAG内容安全审核系统 - 基于领域知识库的检索增强生成系统
依赖安装：
pip install streamlit pandas numpy sentence-transformers openai zhipuai dashscope python-dotenv
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
from openai import OpenAI
import dashscope
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Optional, Tuple
import heapq


# ==================== 配置部分 ====================
class Config:
    VECTOR_DB_PATH = "vector_db.npz"  # 修改为numpy格式
    METADATA_PATH = "metadata.pkl"
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_DIM = 384
    TOP_K = 5
    RERANK_TOP_K = 3

    # LLM配置
    SUPPORTED_MODELS = {
        "OpenAI GPT-4": {"provider": "openai", "model": "gpt-4"},
        "OpenAI GPT-3.5": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "通义千问 Qwen-Plus": {"provider": "dashscope", "model": "qwen-plus"},
        "通义千问 Qwen-Turbo": {"provider": "dashscope", "model": "qwen-turbo"},
        "智谱 GLM-4-FLASH": {"provider": "zhipu", "model": "glm-4-flash"},
        "智谱 GLM-3-Turbo": {"provider": "zhipu", "model": "glm-3-turbo"},
    }


# ==================== 纯numpy实现的向量索引 ====================
class NumpyVectorIndex:
    """纯numpy实现的向量索引，替代faiss"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = np.zeros((0, dimension), dtype=np.float32)
        self.normalized = False
    
    def add(self, vectors: np.ndarray):
        """添加向量到索引"""
        if len(vectors.shape) != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected 2D array of shape (n, {self.dimension}), got {vectors.shape}")
            
        # 如果当前没有向量，直接添加
        if self.vectors.size == 0:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的k个向量
        返回: (distances, indices)
        """
        if self.vectors.size == 0:
            return np.array([]), np.array([])
            
        query = query.astype(np.float32)
        
        # 计算余弦相似度 (内积，因为向量已经归一化)
        similarities = np.dot(query, self.vectors.T)  # shape: (1, n_vectors)
        
        # 获取top-k
        if k >= len(self.vectors):
            k = len(self.vectors)
        
        # 使用堆来获取top-k
        if k <= 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        # np.argpartition 的 kth 是 0-based，必须 < n
        n = similarities.shape[1]
        if k > n:
            k = n

        top_k_indices = np.argpartition(-similarities[0], k - 1)[:k]
        top_k_scores = similarities[0][top_k_indices]
        
        # 按分数排序
        sorted_indices = np.argsort(-top_k_scores)
        return top_k_scores[sorted_indices].reshape(1, -1), top_k_indices[sorted_indices].reshape(1, -1)
    
    def __len__(self) -> int:
        """返回向量数量"""
        return len(self.vectors)
    
    @property
    def ntotal(self) -> int:
        """兼容faiss接口，返回向量数量"""
        return len(self)
    
    def save(self, filepath: str):
        """保存向量到文件"""
        np.savez_compressed(filepath, vectors=self.vectors)
    
    @classmethod
    def load(cls, filepath: str, dimension: int) -> 'NumpyVectorIndex':
        """从文件加载向量"""
        if not os.path.exists(filepath):
            return cls(dimension)
            
        data = np.load(filepath)
        index = cls(dimension)
        index.vectors = data['vectors']
        return index


# ==================== LLM API调用类 ====================
class LLMClient:
    def __init__(self, provider, model, api_key):
        """初始化LLM客户端"""
        self.provider = provider
        self.model = model
        self.api_key = api_key

        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "dashscope":
            dashscope.api_key = api_key
        elif provider == "zhipu":
            self.client = ZhipuAI(api_key=api_key)

    def generate(self, prompt, temperature=0.7, max_tokens=2000):
        """调用LLM生成回答"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的内容安全审核助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            elif self.provider == "dashscope":
                from dashscope import Generation
                response = Generation.call(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的内容安全审核助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    result_format='message'
                )
                if response.status_code == 200:
                    return response.output.choices[0].message.content
                else:
                    return f"API调用失败: {response.message}"

            elif self.provider == "zhipu":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的内容安全审核助手。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"❌ LLM调用错误: {str(e)}\n\n请检查:\n1. API密钥是否正确\n2. 网络连接是否正常\n3. API额度是否充足"


# ==================== 向量数据库管理类 ====================
class VectorDatabase:
    def __init__(self, model_name=Config.EMBEDDING_MODEL):
        """初始化向量数据库"""
        self.model = SentenceTransformer(model_name)
        self.dimension = Config.VECTOR_DIM
        self.index = NumpyVectorIndex(self.dimension)  # 使用纯numpy实现
        self.metadata = []
        self.load_or_create_index()

    def load_or_create_index(self):
        """加载或创建索引"""
        if os.path.exists(Config.VECTOR_DB_PATH) and os.path.exists(Config.METADATA_PATH):
            self.index = NumpyVectorIndex.load(Config.VECTOR_DB_PATH, self.dimension)
            with open(Config.METADATA_PATH, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.index = NumpyVectorIndex(self.dimension)
            self.metadata = []

    def save_index(self):
        """保存索引和元数据"""
        self.index.save(Config.VECTOR_DB_PATH)
        with open(Config.METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

    def build_from_csv(self, csv_path, text_column, category_column=None):
        """从CSV文件构建向量数据库"""
        # Streamlit 上传组件返回的是 UploadedFile；这里兼容 UploadedFile / 文件路径 / 类文件对象
        try:
            if hasattr(csv_path, "getvalue"):
                raw = csv_path.getvalue()
                if not raw or not raw.strip():
                    raise ValueError("上传的CSV为空（文件大小为0或只有空白内容）。请检查导出的CSV是否包含表头和数据。")
                from io import BytesIO
                bio = BytesIO(raw)
                # 优先尝试 utf-8-sig，其次 gbk
                try:
                    df = pd.read_csv(bio, encoding="utf-8-sig")
                except Exception:
                    bio.seek(0)
                    df = pd.read_csv(bio, encoding="gbk")
            else:
                df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            raise ValueError("无法解析CSV：没有读到任何列（No columns to parse）。请确认CSV不是空文件，且第一行包含表头，例如：id,category,content")

        if text_column not in df.columns:
            raise ValueError(f"列 '{text_column}' 不存在于CSV文件中")

        texts = df[text_column].fillna("").tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # 归一化向量（L2范数）
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # 创建新索引并添加向量
        self.index = NumpyVectorIndex(self.dimension)
        self.index.add(embeddings.astype('float32'))

        self.metadata = []
        for idx, row in df.iterrows():
            meta = {
                'id': idx,
                'text': row[text_column],
                'category': row[category_column] if category_column and category_column in df.columns else "未分类",
                'timestamp': datetime.now().isoformat(),
                'other_fields': {k: v for k, v in row.items() if k not in [text_column, category_column]}
            }
            self.metadata.append(meta)

        self.save_index()
        return len(texts)

    def add_document(self, text, category="未分类", **kwargs):
        """添加单个文档"""
        embedding = self.model.encode([text])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        self.index.add(embedding.astype('float32'))

        meta = {
            'id': len(self.metadata),
            'text': text,
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'other_fields': kwargs
        }
        self.metadata.append(meta)
        self.save_index()
        return meta['id']

    def delete_document(self, doc_id):
        """删除文档（通过重建索引）"""
        if doc_id < 0 or doc_id >= len(self.metadata):
            raise ValueError("无效的文档ID")

        self.metadata.pop(doc_id)

        # 更新剩余文档的ID
        for i in range(doc_id, len(self.metadata)):
            self.metadata[i]['id'] = i

        # 重建索引
        if self.metadata:
            texts = [m['text'] for m in self.metadata]
            embeddings = self.model.encode(texts)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index = NumpyVectorIndex(self.dimension)
            self.index.add(embeddings.astype('float32'))
        else:
            self.index = NumpyVectorIndex(self.dimension)

        self.save_index()

    def update_document(self, doc_id, text=None, category=None, **kwargs):
        """更新文档"""
        if doc_id < 0 or doc_id >= len(self.metadata):
            raise ValueError("无效的文档ID")

        if text:
            self.metadata[doc_id]['text'] = text
        if category:
            self.metadata[doc_id]['category'] = category
        if kwargs:
            self.metadata[doc_id]['other_fields'].update(kwargs)

        self.metadata[doc_id]['timestamp'] = datetime.now().isoformat()

        # 重建索引
        texts = [m['text'] for m in self.metadata]
        embeddings = self.model.encode(texts)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index = NumpyVectorIndex(self.dimension)
        self.index.add(embeddings.astype('float32'))

        self.save_index()

    def search(self, query, top_k=Config.TOP_K):
        """语义检索"""
        if len(self.index) == 0:
            return []

        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, len(self.index)))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['score'] = float(dist)
                results.append(result)

        return results

    def get_all_documents(self):
        """获取所有文档"""
        return self.metadata

    def get_statistics(self):
        """获取数据库统计信息"""
        categories = {}
        for meta in self.metadata:
            cat = meta.get('category', '未分类')
            categories[cat] = categories.get(cat, 0) + 1

        return {
            'total_documents': len(self.metadata),
            'categories': categories,
            'index_size': len(self.index) if self.index else 0
        }


# ==================== RAG检索增强生成类 ====================
# [RAGSystem 类保持不变，与原始代码相同]
class RAGSystem:
    def __init__(self, vector_db, llm_client=None):
        self.vector_db = vector_db
        self.llm_client = llm_client

    def set_llm_client(self, llm_client):
        """设置LLM客户端"""
        self.llm_client = llm_client

    def rerank(self, query, results, top_k=Config.RERANK_TOP_K):
        """重排序检索结果"""
        if not results:
            return []

        query_lower = query.lower()

        for result in results:
            text_lower = result['text'].lower()
            keyword_match = sum(1 for word in query_lower.split() if word in text_lower)
            result['rerank_score'] = result['score'] * 0.7 + (keyword_match / max(len(query_lower.split()), 1)) * 0.3

        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        return results[:top_k]

    def generate_context(self, results):
        """生成上下文字符串"""
        if not results:
            return "未找到相关知识。"

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[知识片段{i}] 类别：{result['category']}\n内容：{result['text']}\n相关度：{result.get('rerank_score', result['score']):.3f}")

        return "\n\n".join(context_parts)

    def generate_prompt(self, query, context):
        """生成提示词"""
        prompt = f"""你是一个基于领域知识库的内容安全审核助手。请根据提供的知识库内容回答用户问题。

知识库内容：
{context}

用户问题：{query}

请基于上述知识库内容回答问题。如果知识库中没有相关信息，请明确说明。回答要求：
1. 准确引用知识库内容
2. 保持客观中立
3. 如涉及敏感内容，需要特别谨慎
4. 说明判断依据

回答："""
        return prompt

    def answer(self, query, use_rerank=True, temperature=0.7):
        """回答用户问题"""
        # 检索
        results = self.vector_db.search(query, top_k=Config.TOP_K)

        if not results:
            return {
                'answer': "知识库中未找到相关信息，无法回答该问题。",
                'context': "",
                'retrieved_docs': []
            }

        # 重排序
        if use_rerank:
            results = self.rerank(query, results)
        else:
            results = results[:Config.RERANK_TOP_K]

        # 生成上下文
        context = self.generate_context(results)

        # 生成提示词
        prompt = self.generate_prompt(query, context)

        # 调用LLM生成答案
        if self.llm_client:
            answer = self.llm_client.generate(prompt, temperature=temperature)
        else:
            answer = "⚠️ 未配置LLM API，无法生成答案。请在侧边栏配置API密钥。"

        return {
            'answer': answer,
            'context': context,
            'prompt': prompt,
            'retrieved_docs': results
        }

    def _generate_summary(self, results):
        """生成检索结果摘要"""
        summary_parts = []
        for i, result in enumerate(results, 1):
            summary_parts.append(f"{i}. [{result['category']}] {result['text'][:100]}...")
        return "\n".join(summary_parts)


# ==================== Streamlit Web界面 ====================
# [main 函数保持不变，与原始代码相同]
def main():
    st.set_page_config(page_title="RAG内容安全审核系统", page_icon="🔍", layout="wide")

    st.title("🔍 基于领域知识库的内容安全审核系统")
    st.markdown("---")

    # 初始化
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase()

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(st.session_state.vector_db)

    if 'llm_configured' not in st.session_state:
        st.session_state.llm_configured = False

    # 侧边栏 - 系统管理
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # LLM配置
        with st.expander("🤖 大模型配置", expanded=not st.session_state.llm_configured):
            selected_model = st.selectbox(
                "选择模型",
                list(Config.SUPPORTED_MODELS.keys())
            )

            model_info = Config.SUPPORTED_MODELS[selected_model]

            api_key = st.text_input(
                "API密钥",
                type="password",
                help=f"请输入{selected_model}的API密钥"
            )

            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("温度参数", 0.0, 1.0, 0.7, 0.1)
            with col2:
                if st.button("保存配置", type="primary"):
                    if api_key:
                        try:
                            llm_client = LLMClient(
                                provider=model_info["provider"],
                                model=model_info["model"],
                                api_key=api_key
                            )
                            st.session_state.rag_system.set_llm_client(llm_client)
                            st.session_state.llm_configured = True
                            st.session_state.temperature = temperature
                            st.success("✅ 配置成功！")
                        except Exception as e:
                            st.error(f"❌ 配置失败：{str(e)}")
                    else:
                        st.warning("⚠️ 请输入API密钥")

            if st.session_state.llm_configured:
                st.success(f"✅ 当前模型：{selected_model}")

            # API获取指南
            with st.expander("📖 API密钥获取指南"):
                st.markdown("""
                **OpenAI**
                - 官网：https://platform.openai.com/
                - 注册后在API Keys页面创建

                **通义千问**
                - 官网：https://dashscope.aliyun.com/
                - 阿里云账号登录后获取

                **智谱AI**
                - 官网：https://open.bigmodel.cn/
                - 注册后在个人中心获取
                """)

        st.markdown("---")
        st.header("📊 知识库管理")

        tab1, tab2, tab3 = st.tabs(["构建", "管理", "统计"])

        with tab1:
            st.subheader("从CSV构建")
            uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.write("预览：", df.head(3))

                text_col = st.selectbox("文本列", df.columns.tolist())
                category_col = st.selectbox("分类列", ["无"] + df.columns.tolist())

                if st.button("构建向量数据库", type="primary"):
                    with st.spinner("构建中..."):
                        cat_col = None if category_col == "无" else category_col
                        count = st.session_state.vector_db.build_from_csv(
                            uploaded_file, text_col, cat_col
                        )
                        st.success(f"✅ 导入 {count} 条数据")
                        st.rerun()

        with tab2:
            st.subheader("添加知识")
            new_text = st.text_area("内容", height=100)
            new_category = st.text_input("分类", value="未分类")

            if st.button("➕ 添加"):
                if new_text:
                    doc_id = st.session_state.vector_db.add_document(new_text, new_category)
                    st.success(f"✅ ID: {doc_id}")
                    st.rerun()

            st.subheader("删除知识")
            del_id = st.number_input("文档ID", min_value=0, step=1)
            if st.button("🗑️ 删除"):
                try:
                    st.session_state.vector_db.delete_document(del_id)
                    st.success("✅ 已删除")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {str(e)}")

        with tab3:
            stats = st.session_state.vector_db.get_statistics()
            st.metric("文档总数", stats['total_documents'])
            st.metric("索引大小", stats['index_size'])

            st.subheader("分类分布")
            if stats['categories']:
                for cat, count in stats['categories'].items():
                    st.write(f"- **{cat}**: {count}")

    # 主界面 - RAG问答
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("💬 智能问答")

        if not st.session_state.llm_configured:
            st.warning("⚠️ 请先在侧边栏配置大模型API")

        query = st.text_area("请输入您的问题：", height=120,
                             placeholder="例如：关于某历史事件的准确描述是什么？")

        col_a, col_b = st.columns(2)
        with col_a:
            use_rerank = st.checkbox("启用重排序", value=True)
        with col_b:
            show_prompt = st.checkbox("显示提示词", value=False)

        if st.button("🔍 提交查询", type="primary", use_container_width=True):
            if query:
                if not st.session_state.llm_configured:
                    st.error("❌ 请先配置大模型API")
                else:
                    with st.spinner("🤔 正在思考..."):
                        result = st.session_state.rag_system.answer(
                            query,
                            use_rerank,
                            temperature=st.session_state.get('temperature', 0.7)
                        )

                        st.subheader("📝 系统回答")
                        st.markdown(result['answer'])

                        if show_prompt:
                            with st.expander("📋 查看完整提示词"):
                                st.code(result['prompt'], language="text")

                        with st.expander("🔎 查看检索详情"):
                            st.markdown("**检索到的知识片段：**")
                            for i, doc in enumerate(result['retrieved_docs'], 1):
                                score = doc.get('rerank_score', doc['score'])
                                st.markdown(f"""
                                ---
                                **片段 {i}** | 相关度: `{score:.3f}`
                                - **类别**: {doc['category']}
                                - **内容**: {doc['text'][:200]}{'...' if len(doc['text']) > 200 else ''}
                                """)
            else:
                st.warning("⚠️ 请输入问题")

    with col2:
        st.header("📚 知识库浏览")

        docs = st.session_state.vector_db.get_all_documents()

        if docs:
            search_term = st.text_input("🔍 搜索")

            filtered_docs = docs
            if search_term:
                filtered_docs = [d for d in docs if search_term.lower() in d['text'].lower()]

            st.write(f"显示 {len(filtered_docs)} / {len(docs)} 条")

            for doc in filtered_docs[:20]:
                with st.expander(f"ID: {doc['id']} | {doc['category']}", expanded=False):
                    st.write(f"**内容**: {doc['text']}")
                    st.caption(f"时间: {doc['timestamp']}")
                    if doc.get('other_fields'):
                        st.json(doc['other_fields'])
        else:
            st.info("💡 知识库为空，请先导入数据")


if __name__ == "__main__":
    main()