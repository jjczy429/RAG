import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
import faiss

try:
    from FlagEmbedding import FlagReranker
except Exception:
    FlagReranker = None
from datetime import datetime
import json
from openai import OpenAI
import dashscope
from zhipuai import ZhipuAI
from typing import List, Dict, Any, Optional, Tuple
import heapq
from pathlib import Path
import threading
import tempfile


# ==================== 配置部分 ====================
class Config:
    VECTOR_DB_PATH = "vector_db.faiss"
    METADATA_PATH = "metadata.pkl"
    # 默认使用本地已缓存的轻量模型；如已下载 BGE 系列可在界面中切换
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    TOP_K = 5
    RERANK_TOP_K = 3

    # 各 provider 的 Base URL（写死在代码里；如需走代理/私有化部署请在此修改）
    OPENAI_BASE_URL = "https://api.zhizengzeng.com/v1"
    DASHSCOPE_BASE_URL = ""
    ZHIPU_BASE_URL = ""

    # 重排序（BAAI）配置；默认使用 base 版本，已下载 v2-m3 可在界面中切换
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    RERANK_CANDIDATES = 10
    RERANK_USE_FP16 = True

    # 支持的 Embedding 模型列表（名称 -> HuggingFace model id）
    # 第一项为默认值，请确保对应模型已在本地缓存，否则启动时会尝试联网下载
    SUPPORTED_EMBEDDING_MODELS = {
        "Multilingual-MiniLM（轻量多语言，默认）": "paraphrase-multilingual-MiniLM-L12-v2",
        "BGE-M3（多语言）": "BAAI/bge-m3",
        "BGE-Large-ZH（中文大模型）": "BAAI/bge-large-zh-v1.5",
        "BGE-Base-ZH（中文基础）": "BAAI/bge-base-zh-v1.5",
        "BGE-Small-ZH（中文轻量）": "BAAI/bge-small-zh-v1.5",
        "BGE-Large-EN（英文大模型）": "BAAI/bge-large-en-v1.5",
        "BGE-Base-EN（英文基础）": "BAAI/bge-base-en-v1.5",
    }

    # 支持的 Reranker 模型列表（名称 -> HuggingFace model id）
    # 第一项为默认值，请确保对应模型已在本地缓存
    SUPPORTED_RERANKER_MODELS = {
        "BGE-Reranker-Base（默认）": "BAAI/bge-reranker-base",
        "BGE-Reranker-V2-M3（多语言）": "BAAI/bge-reranker-v2-m3",
        "BGE-Reranker-Large（大模型）": "BAAI/bge-reranker-large",
    }

    # LLM配置
    SUPPORTED_MODELS = {
        "OpenAI GPT-4": {"provider": "openai", "model": "gpt-4"},
        "OpenAI GPT-3.5": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "通义千问 Qwen-Plus": {"provider": "dashscope", "model": "qwen-plus"},
        "通义千问 Qwen-Turbo": {"provider": "dashscope", "model": "qwen-turbo"},
        "智谱 GLM-4-FLASH": {"provider": "zhipu", "model": "glm-4-flash"},
        "智谱 GLM-3-Turbo": {"provider": "zhipu", "model": "glm-3-turbo"},
    }


# ==================== FAISS 向量索引 ====================
class FaissVectorIndex:
    """基于 FAISS 的向量索引（使用内积 + 归一化向量实现余弦相似度）"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vectors: np.ndarray):
        """添加向量到索引"""
        if len(vectors.shape) != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected 2D array of shape (n, {self.dimension}), got {vectors.shape}")

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的 k 个向量
        返回: (distances, indices)
        """
        if self.index.ntotal == 0 or k <= 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        query = np.ascontiguousarray(query.astype(np.float32))
        distances, indices = self.index.search(query, min(k, self.index.ntotal))
        return distances, indices

    def __len__(self) -> int:
        """返回向量数量"""
        return int(self.index.ntotal)

    @property
    def ntotal(self) -> int:
        """兼容接口，返回向量数量"""
        return int(self.index.ntotal)

    def save(self, filepath: str):
        """保存索引到文件"""
        faiss.write_index(self.index, filepath)

    @classmethod
    def load(cls, filepath: str, dimension: int) -> 'FaissVectorIndex':
        """
        从 FAISS 索引文件加载索引。
        如果文件不存在或加载失败，则返回空索引。
        """
        index = cls(dimension)

        if not os.path.exists(filepath):
            return index

        try:
            index.index = faiss.read_index(filepath)
        except Exception:
            # 如果读取失败，则返回一个空的 FAISS 索引
            index.index = faiss.IndexFlatIP(dimension)

        return index


# ==================== API Key 本地持久化 ====================
class ApiKeyStore:
    """本地保存/读取各 provider 的 API Key。

    说明：
    - 默认保存在用户目录下的 .rag_api_keys.json（不建议提交到仓库）
    - 也可通过环境变量 RAG_API_KEYS_PATH 指定路径
    """

    @staticmethod
    def _default_path() -> Path:
        env = os.environ.get("RAG_API_KEYS_PATH")
        if env:
            return Path(env).expanduser()
        return Path.home() / ".rag_api_keys.json"

    @classmethod
    def load_all(cls) -> Dict[str, str]:
        path = cls._default_path()
        try:
            if not path.exists():
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v}
            return {}
        except Exception:
            return {}

    @classmethod
    def save_key(cls, provider: str, api_key: str) -> None:
        path = cls._default_path()
        data = cls.load_all()
        data[str(provider)] = str(api_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def get_key(cls, provider: str) -> str:
        return cls.load_all().get(str(provider), "")


# ==================== LLM API调用类 ====================
class LLMClient:
    def __init__(self, provider, model, api_key, base_url: str = ""):
        """初始化LLM客户端"""
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "").strip()

        if provider == "openai":
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=api_key)
        elif provider == "dashscope":
            dashscope.api_key = api_key
            # dashscope SDK 通常不需要 base_url；保留配置位以便后续扩展
        elif provider == "zhipu":
            try:
                if self.base_url:
                    self.client = ZhipuAI(api_key=api_key, base_url=self.base_url)
                else:
                    self.client = ZhipuAI(api_key=api_key)
            except TypeError:
                # 兼容旧版 SDK（不支持 base_url 参数）
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
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # 从模型自身获取向量维度，不依赖固定配置
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = FaissVectorIndex(self.dimension)
        self.metadata = []

        # 绑定到某个 CSV 文件后，增删改将同步写回该 CSV
        self.csv_path: Optional[str] = None
        self.csv_text_column: Optional[str] = None
        self.csv_category_column: Optional[str] = None
        self.csv_other_columns: List[str] = []
        # 保存原始 CSV 列顺序（例如: ["id", "category", "content"]）
        self.csv_column_order: List[str] = []
        self._csv_lock = threading.Lock()

        self.load_or_create_index()

    def load_or_create_index(self):
        """加载或创建索引；若已有索引与当前模型维度不匹配则自动清空"""
        if os.path.exists(Config.VECTOR_DB_PATH) and os.path.exists(Config.METADATA_PATH):
            loaded = FaissVectorIndex.load(Config.VECTOR_DB_PATH, self.dimension)
            # 维度一致才复用，否则丢弃旧索引（说明模型已切换）
            if loaded.index.d == self.dimension:
                self.index = loaded
                with open(Config.METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.index = FaissVectorIndex(self.dimension)
                self.metadata = []
        else:
            self.index = FaissVectorIndex(self.dimension)
            self.metadata = []

    def save_index(self):
        """保存索引和元数据"""
        self.index.save(Config.VECTOR_DB_PATH)
        with open(Config.METADATA_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

    def bind_csv(self, csv_path: str, text_column: str, category_column: Optional[str] = None,
                 other_columns: Optional[List[str]] = None, column_order: Optional[List[str]] = None) -> None:
        self.csv_path = str(csv_path) if csv_path else None
        self.csv_text_column = str(text_column) if text_column else None
        self.csv_category_column = str(category_column) if category_column else None
        self.csv_other_columns = list(other_columns or [])
        # 记录原始列顺序，用于写回时保持列顺序不变
        self.csv_column_order = list(column_order or [])

    def _write_back_csv(self) -> None:
        if not self.csv_path:
            return
        if not self.csv_text_column:
            return

        with self._csv_lock:
            rows = []
            for m in self.metadata:
                row = {}
                row[self.csv_text_column] = m.get('text', '')
                if self.csv_category_column:
                    row[self.csv_category_column] = m.get('category', '未分类')
                other = m.get('other_fields') or {}
                for k in self.csv_other_columns:
                    row[k] = other.get(k, '')
                rows.append(row)

            df = pd.DataFrame(rows)

            # 如果记录了原始列顺序，则按原顺序重排列（例如: id, category, content）
            if self.csv_column_order:
                # 只保留当前存在的列，并保持原有顺序
                ordered_cols = [c for c in self.csv_column_order if c in df.columns]
                # 如有新增列（原顺序中没有的），追加在末尾，避免丢失
                remaining_cols = [c for c in df.columns if c not in ordered_cols]
                df = df[ordered_cols + remaining_cols]

            # Windows 下目标文件被占用（例如被 Excel 打开）时，os.replace 会抛 PermissionError。
            # 这里采用“写入同名文件”的方式，并加简单重试，尽量提高成功率。
            import time

            last_err = None
            for _ in range(5):
                try:
                    df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
                    last_err = None
                    break
                except PermissionError as e:
                    last_err = e
                    time.sleep(0.2)

            if last_err is not None:
                raise last_err

    def build_from_csv(self, csv_path, text_column, category_column=None, bind_to_csv_path: Optional[str] = None):
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
        self.index = FaissVectorIndex(self.dimension)
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

        if bind_to_csv_path:
            other_cols = [c for c in df.columns.tolist() if c not in [text_column] + ([category_column] if category_column else [])]
            # 传入原始列顺序（例如: ["id", "category", "content"]），用于写回时保持列顺序
            self.bind_csv(bind_to_csv_path, text_column, category_column,
                          other_columns=other_cols, column_order=df.columns.tolist())

        self.save_index()
        self._write_back_csv()
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
        self._write_back_csv()
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
            self.index = FaissVectorIndex(self.dimension)
            self.index.add(embeddings.astype('float32'))
        else:
            self.index = FaissVectorIndex(self.dimension)

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
        self.index = FaissVectorIndex(self.dimension)
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
    def __init__(self, vector_db, llm_client=None, reranker_model: str = Config.RERANKER_MODEL, rerank_use_fp16: bool = Config.RERANK_USE_FP16):
        self.vector_db = vector_db
        self.llm_client = llm_client
        self.reranker_model = reranker_model
        self.rerank_use_fp16 = rerank_use_fp16
        self._reranker = None

    def set_llm_client(self, llm_client):
        """设置LLM客户端"""
        self.llm_client = llm_client

    def _get_reranker(self):
        if self._reranker is not None:
            return self._reranker

        if FlagReranker is None:
            self._reranker = None
            return None

        try:
            self._reranker = FlagReranker(self.reranker_model, use_fp16=self.rerank_use_fp16)
        except Exception:
            self._reranker = None
        return self._reranker

    def rerank(self, query, results, top_k: int = Config.RERANK_TOP_K):
        """使用 BAAI reranker 对检索结果重排"""
        if not results:
            return []

        # 允许前端传 0 表示不返回任何重排结果（相当于禁用重排输出）
        if top_k <= 0:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            return results[:top_k]

        pairs = [[query, r.get('text', '')] for r in results]
        try:
            scores = reranker.compute_score(pairs)
        except Exception:
            return results[:top_k]

        for r, s in zip(results, scores):
            r['rerank_score'] = float(s)

        results.sort(key=lambda x: x.get('rerank_score', x.get('score', 0.0)), reverse=True)
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
        prompt = f"""{query}

{context}"""
        return prompt

    def answer(self, query, use_rerank=True, temperature=0.7, rerank_top_k: Optional[int] = None):
        """回答用户问题"""
        # 先检索较多候选，再重排截断
        candidates_k = max(Config.TOP_K, Config.RERANK_CANDIDATES)
        results = self.vector_db.search(query, top_k=candidates_k)

        if not results:
            return {
                'answer': "知识库中未找到相关信息，无法回答该问题。",
                'context': "",
                'retrieved_docs': []
            }

        # 前端可控重排数量（0-5）。None 时沿用默认配置。
        if rerank_top_k is None:
            rerank_top_k = Config.RERANK_TOP_K
        # 范围夹紧，避免异常输入
        rerank_top_k = max(0, min(int(rerank_top_k), 5))

        if use_rerank:
            results = self.rerank(query, results, top_k=rerank_top_k)
        else:
            results = results[:rerank_top_k]

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

    # 初始化 session_state 默认值
    if 'embedding_model_name' not in st.session_state:
        st.session_state.embedding_model_name = Config.EMBEDDING_MODEL
    if 'reranker_model_name' not in st.session_state:
        st.session_state.reranker_model_name = Config.RERANKER_MODEL
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = VectorDatabase(st.session_state.embedding_model_name)
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(
            st.session_state.vector_db,
            reranker_model=st.session_state.reranker_model_name
        )
    if 'llm_configured' not in st.session_state:
        st.session_state.llm_configured = False

    # 侧边栏 - 系统管理
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # Embedding 模型 & Reranker 模型选择
        with st.expander("🧩 向量/重排序模型配置", expanded=False):
            emb_model_label = st.selectbox(
                "Embedding 模型",
                list(Config.SUPPORTED_EMBEDDING_MODELS.keys()),
                index=list(Config.SUPPORTED_EMBEDDING_MODELS.values()).index(
                    st.session_state.embedding_model_name
                ) if st.session_state.embedding_model_name in Config.SUPPORTED_EMBEDDING_MODELS.values() else 0,
                help="选择文本向量化模型。更换模型后需重新构建知识库。"
            )
            selected_emb_model = Config.SUPPORTED_EMBEDDING_MODELS[emb_model_label]

            reranker_model_label = st.selectbox(
                "Reranker 重排序模型",
                list(Config.SUPPORTED_RERANKER_MODELS.keys()),
                index=list(Config.SUPPORTED_RERANKER_MODELS.values()).index(
                    st.session_state.reranker_model_name
                ) if st.session_state.reranker_model_name in Config.SUPPORTED_RERANKER_MODELS.values() else 0,
                help="选择重排序模型，用于提升检索精度。"
            )
            selected_reranker_model = Config.SUPPORTED_RERANKER_MODELS[reranker_model_label]

            if st.button("应用模型配置", type="primary"):
                emb_changed = selected_emb_model != st.session_state.embedding_model_name
                reranker_changed = selected_reranker_model != st.session_state.reranker_model_name

                if emb_changed:
                    with st.spinner(f"正在加载 Embedding 模型：{selected_emb_model} …"):
                        try:
                            new_db = VectorDatabase(selected_emb_model)
                            st.session_state.vector_db = new_db
                            st.session_state.embedding_model_name = selected_emb_model
                            st.session_state.rag_system = RAGSystem(
                                new_db,
                                reranker_model=selected_reranker_model
                            )
                            st.session_state.reranker_model_name = selected_reranker_model
                            st.success(f"✅ Embedding 模型已切换为：{selected_emb_model}")
                            if new_db.index.ntotal == 0:
                                st.warning("⚠️ 模型已切换，旧索引维度不匹配已清空，请重新构建知识库。")
                        except Exception as e:
                            st.error(f"❌ 模型加载失败：{e}")
                elif reranker_changed:
                    st.session_state.rag_system.reranker_model = selected_reranker_model
                    st.session_state.rag_system._reranker = None  # 清空缓存，下次使用时重新加载
                    st.session_state.reranker_model_name = selected_reranker_model
                    st.success(f"✅ Reranker 模型已切换为：{selected_reranker_model}")
                else:
                    st.info("模型配置未发生变化。")

            st.caption(f"当前 Embedding：`{st.session_state.embedding_model_name}`")
            st.caption(f"当前 Reranker：`{st.session_state.reranker_model_name}`")

        # LLM配置
        with st.expander("🤖 大模型配置", expanded=not st.session_state.llm_configured):
            selected_model = st.selectbox(
                "选择模型",
                list(Config.SUPPORTED_MODELS.keys())
            )

            model_info = Config.SUPPORTED_MODELS[selected_model]

            saved_key = ApiKeyStore.get_key(model_info["provider"])

            api_key = st.text_input(
                "API密钥",
                type="password",
                value=saved_key,
                help=f"请输入{selected_model}的API密钥（会在本机保存，后续自动填充）"
            )

            base_url = ""
            if model_info["provider"] == "openai":
                base_url = Config.OPENAI_BASE_URL
            elif model_info["provider"] == "dashscope":
                base_url = Config.DASHSCOPE_BASE_URL
            elif model_info["provider"] == "zhipu":
                base_url = Config.ZHIPU_BASE_URL

            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("温度参数", 0.0, 1.0, 0.7, 0.1)
            with col2:
                if st.button("保存配置", type="primary"):
                    if api_key:
                        try:
                            # 保存API Key到本地，后续运行自动填充
                            ApiKeyStore.save_key(model_info["provider"], api_key)

                            llm_client = LLMClient(
                                provider=model_info["provider"],
                                model=model_info["model"],
                                api_key=api_key,
                                base_url=base_url
                            )
                            st.session_state.rag_system.set_llm_client(llm_client)
                            st.session_state.llm_configured = True
                            st.session_state.temperature = temperature
                            st.success("✅ 配置已保存！")
                        except Exception as e:
                            st.error(f"❌ 配置失败：{str(e)}")
                    else:
                        st.warning("⚠️ 请输入API密钥")

            if st.session_state.llm_configured:
                st.success(f"✅ 当前模型：{selected_model}")


        st.markdown("---")
        st.header("📊 知识库管理")

        tab1, tab2, tab3 = st.tabs(["构建", "管理", "统计"])

        with tab1:
            st.subheader("从CSV文件构建")

            uploaded_file = st.file_uploader(
                "上传 CSV 知识库文件",
                type=["csv"],
                help="支持 UTF-8 / UTF-8-BOM / GBK 编码的 CSV 文件"
            )

            if uploaded_file is not None:
                try:
                    raw = uploaded_file.getvalue()
                    from io import BytesIO
                    try:
                        df_preview = pd.read_csv(BytesIO(raw), encoding="utf-8-sig")
                    except Exception:
                        df_preview = pd.read_csv(BytesIO(raw), encoding="gbk")
                    st.write("文件预览（前3行）：", df_preview.head(3))
                except Exception as e:
                    st.error(f"❌ 文件读取失败：{e}")
                    df_preview = None
            else:
                df_preview = None
                st.info("💡 请上传 CSV 文件以构建或刷新知识库。")

            if df_preview is not None:
                text_col = st.selectbox("文本列", df_preview.columns.tolist())
                category_col = st.selectbox("分类列", ["无"] + df_preview.columns.tolist())

                if st.button("构建/刷新向量数据库", type="primary"):
                    with st.spinner("构建中，请稍候…"):
                        try:
                            cat_col = None if category_col == "无" else category_col
                            count = st.session_state.vector_db.build_from_csv(
                                uploaded_file, text_col, cat_col
                            )
                            st.success(f"✅ 成功从「{uploaded_file.name}」导入 {count} 条数据")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ 构建失败：{e}")

        with tab2:
            st.subheader("添加知识")
            new_text = st.text_area("内容", height=100)
            new_category = st.text_input("分类", value="未分类")

            if st.button("➕ 添加"):
                if new_text:
                    doc_id = st.session_state.vector_db.add_document(new_text, new_category)
                    st.success(f"✅ ID: {doc_id}")
                    st.rerun()

            st.subheader("编辑知识")
            docs_for_edit = st.session_state.vector_db.get_all_documents()
            if docs_for_edit:
                edit_id = st.number_input("文档ID（编辑）", min_value=0, max_value=max(len(docs_for_edit) - 1, 0), step=1)
                current = docs_for_edit[int(edit_id)]
                edit_text = st.text_area("内容（编辑）", value=str(current.get('text', '')), height=120)
                edit_category = st.text_input("分类（编辑）", value=str(current.get('category', '未分类')))

                if st.button("✏️ 保存编辑"):
                    try:
                        st.session_state.vector_db.update_document(int(edit_id), text=edit_text, category=edit_category)
                        st.success("✅ 已保存（并同步写回CSV）")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("💡 知识库为空，无法编辑")

            st.subheader("删除知识")
            docs_for_del = st.session_state.vector_db.get_all_documents()
            if docs_for_del:
                del_id = st.number_input("文档ID（删除）", min_value=0, max_value=max(len(docs_for_del) - 1, 0), step=1)
                if st.button("🗑️ 删除"):
                    try:
                        st.session_state.vector_db.delete_document(int(del_id))
                        st.success("✅ 已删除（并同步写回CSV）")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("💡 知识库为空，无法删除")

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

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            use_rerank = st.checkbox("启用重排序", value=True)
        with col_b:
            show_prompt = st.checkbox("显示提示词", value=False)
        with col_c:
            rerank_top_k = st.number_input("重排数量", min_value=0, max_value=5, value=int(Config.RERANK_TOP_K), step=1)

        if st.button("🔍 提交查询", type="primary", use_container_width=True):
            if query:
                if not st.session_state.llm_configured:
                    st.error("❌ 请先配置大模型API")
                else:
                    with st.spinner("🤔 正在思考..."):
                        result = st.session_state.rag_system.answer(
                            query,
                            use_rerank,
                            temperature=st.session_state.get('temperature', 0.7),
                            rerank_top_k=rerank_top_k
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
