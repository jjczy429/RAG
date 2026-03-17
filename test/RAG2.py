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

# 条件导入 - Gemini 和 Claude SDK
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    import anthropic
except Exception:
    anthropic = None


# ==================== 配置部分 ====================
class Config:
    VECTOR_DB_PATH = "vector_db.faiss"
    METADATA_PATH = "metadata.pkl"
    # 默认使用本地已缓存的轻量模型；如已下载 BGE 系列可在界面中切换
    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
    TOP_K = 5
    RERANK_TOP_K = 3

    # 各 provider 的 Base URL（写死在代码里；如需走代理/私有化部署请在此修改）
    OPENAI_BASE_URL = "https://api.zhizengzeng.com/v1"
    DASHSCOPE_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    ZHIPU_BASE_URL = ""

    # 重排序（BAAI）配置；默认使用 base 版本，已下载 v2-m3 可在界面中切换
    RERANKER_MODEL = "BAAI/bge-reranker-base"
    RERANK_CANDIDATES = 10
    RERANK_USE_FP16 = True

    # 支持的 Embedding 模型列表（名称 -> HuggingFace model id）
    # 第一项为默认值，请确保对应模型已在本地缓存，否则启动时会尝试联网下载
    SUPPORTED_EMBEDDING_MODELS = {
        "Multilingual-MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "BGE-M3": "BAAI/bge-m3",
        "BGE-Large-ZH": "BAAI/bge-large-zh-v1.5",
        "BGE-Base-ZH": "BAAI/bge-base-zh-v1.5",
        "BGE-Small-ZH": "BAAI/bge-small-zh-v1.5",
    }

    # 支持的 Reranker 模型列表（名称 -> HuggingFace model id）
    # 第一项为默认值，请确保对应模型已在本地缓存
    SUPPORTED_RERANKER_MODELS = {
        "BGE-Reranker-Base": "BAAI/bge-reranker-base",
        "BGE-Reranker-V2-M3": "BAAI/bge-reranker-v2-m3",
        "BGE-Reranker-Large": "BAAI/bge-reranker-large",
    }

    # 违规类型列表（用于知识库添加和编辑）
    # 如需修改违规类型，请在此处统一修改
    VIOLATION_TYPES = [
        "无",
        "涉台",
        "涉港",
        "涉疆",
        "涉藏",
        "涉军",
        "涉领导人",
        "侮辱英烈",
        "历史虚无主义",
        "分裂国家",
        "煽动颠覆国家政权",
        "损害国家形象",
        "破坏民族团结",
        "损害国家主权与领土完整",
        "邪教组织",
        "其他违规"
    ]

    # LLM配置
    SUPPORTED_MODELS = {
        "OpenAI GPT-4": {"provider": "openai", "model": "gpt-4"},
        "OpenAI GPT-3.5": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "通义千问 Qwen-Plus": {"provider": "dashscope", "model": "qwen-plus"},
        "通义千问 Qwen-Turbo": {"provider": "dashscope", "model": "qwen-turbo"},
        "智谱 GLM-4-FLASH": {"provider": "zhipu", "model": "glm-4-flash"},
        "智谱 GLM-3-Turbo": {"provider": "zhipu", "model": "glm-3-turbo"},
        "Google Gemini 2.0 Flash": {"provider": "gemini", "model": "gemini-2.0-flash"},
        "Google Gemini 2.5 Pro": {"provider": "gemini", "model": "gemini-2.5-pro"},
        "Google Gemini 1.5 Flash": {"provider": "gemini", "model": "gemini-1.5-flash"},
        "Anthropic Claude 3.5 Sonnet": {"provider": "claude", "model": "claude-3-5-sonnet-20241022"},
        "Anthropic Claude 3 Opus": {"provider": "claude", "model": "claude-3-opus-20240229"},
        "Anthropic Claude 3 Sonnet": {"provider": "claude", "model": "claude-3-opus-20240229"},
    }

    # Gemini 和 Claude 的 Base URL
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    CLAUDE_BASE_URL = "https://api.anthropic.com/v1"


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
        elif provider == "zhipu":
            try:
                if self.base_url:
                    self.client = ZhipuAI(api_key=api_key, base_url=self.base_url)
                else:
                    self.client = ZhipuAI(api_key=api_key)
            except TypeError:
                self.client = ZhipuAI(api_key=api_key)
        elif provider == "gemini":
            if genai is None:
                raise ImportError("请安装 google-generativeai 包: pip install google-generativeai")
            genai.configure(api_key=api_key)
            self.client = genai
        elif provider == "claude":
            if anthropic is None:
                raise ImportError("请安装 anthropic 包: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)

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

            elif self.provider == "gemini":
                model = self.client.GenerativeModel(
                    self.model,
                    system_instruction="你是一个专业的内容安全审核助手。"
                )
                response = model.generate_content(
                    contents=prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )
                return response.text

            elif self.provider == "claude":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system="你是一个专业的内容安全审核助手。",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

        except Exception as e:
            return f"❌ LLM调用错误: {str(e)}\n\n请检查:\n1. API密钥是否正确\n2. 网络连接是否正常\n3. API额度是否充足"


# ==================== 缓存模型（避免每次重新加载） ====================
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    """缓存 SentenceTransformer 模型，避免每次重新加载"""
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner=False)
def get_reranker_model(model_name: str, use_fp16: bool = False):
    """缓存 FlagReranker 模型，避免每次重新加载"""
    if FlagReranker is None:
        return None
    return FlagReranker(model_name, use_fp16=use_fp16)


# ==================== 向量数据库管理类 ====================
class VectorDatabase:
    def __init__(self, model_name=Config.EMBEDDING_MODEL):
        """初始化向量数据库"""
        self.model_name = model_name
        # 使用缓存的模型，避免每次重新加载
        self.model = get_embedding_model(model_name)
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

                # 保存新增的字段
                extra_fields = ['id', 'title', 'violation_type', 'severity', 'suggestion', 'sources']
                for field in extra_fields:
                    if field in m:
                        row[field] = m[field]

                # 其他额外字段
                other = m.get('other_fields') or {}
                for k in self.csv_other_columns:
                    if k not in row:
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
                # 尝试多种编码：utf-8-sig, gbk, gb18030
                for encoding in ['utf-8-sig', 'gbk', 'gb18030', 'utf-8']:
                    try:
                        df = pd.read_csv(bio, encoding=encoding)
                        break
                    except Exception:
                        bio.seek(0)
                        continue
            else:
                # 本地文件也尝试多种编码
                for encoding in ['utf-8-sig', 'gbk', 'gb18030', 'utf-8']:
                    try:
                        df = pd.read_csv(csv_path, encoding=encoding)
                        break
                    except Exception:
                        continue
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
            # 将所有字段都提取出来，方便后续使用
            # 如果CSV中没有id列，则生成 KB- 格式的ID
            csv_id = row.get('id', None)
            if csv_id is not None and csv_id != '':
                doc_id = str(csv_id)
            else:
                doc_id = f"KB-{idx + 1}"

            meta = {
                'id': doc_id,
                'text': row[text_column],
                'category': row.get('category', row.get('category_column', '未分类')) if category_column and category_column in df.columns else "未分类",
                'title': row.get('title', ''),
                'violation_type': row.get('violation_type', row.get('violation_type', '无')),
                'severity': row.get('severity', row.get('severity', '无')),
                'suggestion': row.get('suggestion', row.get('suggestion', '无')),
                'sources': row.get('sources', row.get('sources', '')),
                'timestamp': datetime.now().isoformat(),
            }
            self.metadata.append(meta)

        if bind_to_csv_path:
            other_cols = [c for c in df.columns.tolist() if c not in [text_column] + ([category_column] if category_column else [])]
            # 传入原始列顺序（例如: ["id", "category", "content"]），用于写回时保持列顺序
            self.bind_csv(bind_to_csv_path, text_column, category_column,
                          other_columns=other_cols, column_order=df.columns.tolist())

        self.save_index()
        return len(texts)

    def add_document(self, text, category="未分类", title="", violation_type="无", severity="无", suggestion="无", sources="", **kwargs):
        """添加单个文档"""
        embedding = self.model.encode([text])
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        self.index.add(embedding.astype('float32'))

        # 生成新ID，格式为 KB-{最后一条ID+1}
        if self.metadata:
            last_id = self.metadata[-1].get('id', '')
            # 尝试解析ID中的数字部分
            if isinstance(last_id, str) and last_id.startswith('KB-'):
                try:
                    last_num = int(last_id.split('-')[-1])
                    new_id = f"KB-{last_num + 1}"
                except (ValueError, IndexError):
                    new_id = f"KB-{len(self.metadata)}"
            else:
                new_id = f"KB-{len(self.metadata)}"
        else:
            new_id = "KB-1"

        meta = {
            'id': new_id,
            'text': text,
            'category': category,
            'title': title,
            'violation_type': violation_type,
            'severity': severity,
            'suggestion': suggestion,
            'sources': sources,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        self.metadata.append(meta)
        self.save_index()
        return meta['id']

    def delete_document(self, doc_id):
        """删除文档（通过重建索引）"""
        if doc_id < 0 or doc_id >= len(self.metadata):
            raise ValueError("无效的文档ID")

        self.metadata.pop(doc_id)

        # 重新生成所有文档的ID（格式：KB-{序号+1}）
        for i in range(len(self.metadata)):
            self.metadata[i]['id'] = f"KB-{i + 1}"

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

    def update_document(self, doc_id, text=None, category=None, title=None, violation_type=None, severity=None, suggestion=None, sources=None, **kwargs):
        """更新文档"""
        if doc_id < 0 or doc_id >= len(self.metadata):
            raise ValueError("无效的文档ID")

        # 检查文本是否改变
        text_changed = False
        old_text = self.metadata[doc_id].get('text', '')
        if text and text != old_text:
            text_changed = True
            self.metadata[doc_id]['text'] = text
        if category:
            self.metadata[doc_id]['category'] = category
        if title is not None:
            self.metadata[doc_id]['title'] = title
        if violation_type is not None:
            self.metadata[doc_id]['violation_type'] = violation_type
        if severity is not None:
            self.metadata[doc_id]['severity'] = severity
        if suggestion is not None:
            self.metadata[doc_id]['suggestion'] = suggestion
        if sources is not None:
            self.metadata[doc_id]['sources'] = sources
        if kwargs:
            if 'other_fields' not in self.metadata[doc_id]:
                self.metadata[doc_id]['other_fields'] = {}
            self.metadata[doc_id]['other_fields'].update(kwargs)

        self.metadata[doc_id]['timestamp'] = datetime.now().isoformat()

        # 只在文本改变时才重建索引
        if text_changed:
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

        # 使用缓存的模型，避免每次重新加载
        self._reranker = get_reranker_model(self.reranker_model, self.rerank_use_fp16)
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
        """生成上下文字符串（传递给LLM的完整知识条目）"""
        if not results:
            return "未找到相关知识。"

        context_parts = []
        for i, result in enumerate(results, 1):
            # 传递完整的知识条目，包括违规类型、严重程度、处理建议、来源
            context_parts.append(
                f"""[知识片段{i}]
分类：{result.get('category', '未分类')}
标题：{result.get('title', '')}
内容：{result.get('text', '')}
违规类型：{result.get('violation_type', '无')}
严重程度：{result.get('severity', '无')}
处理建议：{result.get('suggestion', '无')}
参考来源：{result.get('sources', '无')}
相关度得分：{result.get('rerank_score', result.get('score', 0)):.3f}""")

        return "\n\n".join(context_parts)

    def generate_audit_prompt(self, query, context):
        """生成内容审核专用的提示词"""
        prompt = f"""你是一个专业的内容安全审核专家。请根据提供的领域知识库对待审核文本进行判断。

【领域知识库参考】
{context}

【待审核文本】
{query}

【审核要求】
1. 根据知识库中的内容进行判断：
   - 如果知识库中该内容标注为"无违规"，则审核结果为"合规"
   - 如果知识库中该内容标注了违规类型，则审核结果为"不合规"
   - 如果知识库中没有完全匹配的内容，但根据知识库中的类似案例可能违规，标注"需人工复核"
2. 严重程度参考：高 > 中 > 低
3. 必须引用知识库中的相关法律依据和来源

【输出格式】（严格按此格式输出，每项一行）
审核结果：[合规/不合规/需人工复核]
违规类型：[具体违规类型，如无可填"无"]
严重程度：[高/中/低/无]
违规原因：[详细说明违规原因，要引用知识库中的依据]
处理建议：[具体、可操作的处理建议]
参考来源：[知识库中的来源]"""
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

        # 生成提示词（通用问答模式）
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

    def audit_content(self, query, use_rerank=True, temperature=0.3, rerank_top_k: Optional[int] = None):
        """内容安全审核"""
        # 检索更多候选以提高审核准确度
        candidates_k = max(Config.TOP_K * 2, Config.RERANK_CANDIDATES)
        results = self.vector_db.search(query, top_k=candidates_k)

        if not results:
            return {
                'answer': "知识库中未找到相关信息，无法进行准确审核。建议：人工复核。",
                'context': "",
                'retrieved_docs': [],
                'audit_result': {
                    'result': '需人工复核',
                    'violation_type': '无',
                    'severity': '无',
                    'reason': '知识库中未找到相关内容',
                    'suggestion': '请进行人工审核确认',
                    'sources': '无'
                }
            }

        # 前端可控重排数量
        if rerank_top_k is None:
            rerank_top_k = Config.RERANK_TOP_K
        rerank_top_k = max(0, min(int(rerank_top_k), 5))

        if use_rerank:
            results = self.rerank(query, results, top_k=rerank_top_k)
        else:
            results = results[:rerank_top_k]

        # 生成上下文
        context = self.generate_context(results)

        # 生成审核专用提示词
        prompt = self.generate_audit_prompt(query, context)

        # 调用LLM生成审核结果（使用较低温度以提高准确性）
        if self.llm_client:
            answer = self.llm_client.generate(prompt, temperature=temperature)
            # 解析审核结果
            audit_result = self._parse_audit_result(answer)
        else:
            answer = "⚠️ 未配置LLM API，无法生成审核结果。请在侧边栏配置API密钥。"
            audit_result = {
                'result': '需人工复核',
                'violation_type': '无',
                'severity': '无',
                'reason': 'LLM未配置',
                'suggestion': '请先配置API密钥',
                'sources': '无'
            }

        return {
            'answer': answer,
            'context': context,
            'prompt': prompt,
            'retrieved_docs': results,
            'audit_result': audit_result
        }

    def _parse_audit_result(self, answer):
        """解析LLM返回的审核结果"""
        result = {
            'result': '需人工复核',
            'violation_type': '无',
            'severity': '无',
            'reason': '解析失败',
            'suggestion': '请查看完整审核结果',
            'sources': '无'
        }

        try:
            lines = answer.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('审核结果：') or line.startswith('审核结果:'):
                    result['result'] = line.split('：')[-1].split(':')[-1].strip()
                elif line.startswith('违规类型：') or line.startswith('违规类型:'):
                    result['violation_type'] = line.split('：')[-1].split(':')[-1].strip()
                elif line.startswith('严重程度：') or line.startswith('严重程度:'):
                    result['severity'] = line.split('：')[-1].split(':')[-1].strip()
                elif line.startswith('违规原因：') or line.startswith('违规原因:'):
                    result['reason'] = line.split('：')[-1].split(':')[-1].strip()
                elif line.startswith('处理建议：') or line.startswith('处理建议:'):
                    result['suggestion'] = line.split('：')[-1].split(':')[-1].strip()
                elif line.startswith('参考来源：') or line.startswith('参考来源:'):
                    result['sources'] = line.split('：')[-1].split(':')[-1].strip()
        except Exception:
            pass

        return result

    def generate_prompt(self, query, context):
        """生成通用问答提示词"""
        prompt = f"""请根据以下知识库内容回答用户问题。

【知识库参考】
{context}

【用户问题】
{query}

请根据知识库内容给出准确、专业的回答。如果知识库中没有相关信息，请如实说明。"""
        return prompt

    def _generate_summary(self, results):
        """生成检索结果摘要"""
        summary_parts = []
        for i, result in enumerate(results, 1):
            summary_parts.append(f"{i}. [{result['category']}] {result['text'][:100]}...")
        return "\n".join(summary_parts)


# ==================== Streamlit Web界面 ====================
# [main 函数保持不变，与原始代码相同]
def main():
    st.set_page_config(page_title="内容安全审核系统", page_icon="🛡️", layout="wide")

    # 自定义CSS使侧边栏可滚动
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        overflow-y: auto;
        height: 100vh;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 6px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🛡️ 基于领域知识库的内容安全审核系统")
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
            elif model_info["provider"] == "gemini":
                base_url = Config.GEMINI_BASE_URL
            elif model_info["provider"] == "claude":
                base_url = Config.CLAUDE_BASE_URL

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
                    # 尝试多种编码
                    for enc in ['utf-8-sig', 'gbk', 'gb18030', 'utf-8']:
                        try:
                            df_preview = pd.read_csv(BytesIO(raw), encoding=enc)
                            break
                        except Exception:
                            continue
                    else:
                        # 所有编码都失败
                        df_preview = None
                        st.error("❌ 文件编码无法识别，请检查文件编码")
                    if df_preview is not None:
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

            # 新建两列布局，使表单更紧凑
            col_add1, col_add2 = st.columns(2)
            with col_add1:
                new_title = st.text_input("标题", placeholder="请输入知识标题")
                new_category = st.selectbox("分类", ["法律法规", "政策文件", "权威表述", "违规案例", "其他"], index=0)
            with col_add2:
                st.markdown("<br>", unsafe_allow_html=True)  # 占位，保持两列布局

            col_add3, col_add4 = st.columns(2)
            with col_add3:
                new_severity = st.selectbox("严重程度", ["无", "低", "中", "高"], index=0)
            with col_add4:
                new_sources = st.text_input("来源", placeholder="请输入参考来源")

            # 违规类型多选
            new_violation_type = st.multiselect(
                "违规类型（可多选）",
                Config.VIOLATION_TYPES,
                default=["无"],
                help="选择一种或多种违规类型"
            )
            # 将列表转换为逗号分隔的字符串存储
            new_violation_type_str = ", ".join(new_violation_type) if new_violation_type else "无"
            new_text = st.text_area("内容", height=120, placeholder="请输入知识内容")
            new_suggestion = st.text_input("处理建议", placeholder="请输入处理建议")

            if st.button("➕ 添加知识", type="primary"):
                if new_text:
                    try:
                        doc_id = st.session_state.vector_db.add_document(
                            text=new_text,
                            category=new_category,
                            title=new_title,
                            violation_type=new_violation_type_str,
                            severity=new_severity,
                            suggestion=new_suggestion,
                            sources=new_sources
                        )
                        st.success(f"✅ 知识添加成功！ID: {doc_id}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 添加失败：{str(e)}")
                else:
                    st.warning("⚠️ 请输入内容")

            st.subheader("编辑知识")
            docs_for_edit = st.session_state.vector_db.get_all_documents()
            if docs_for_edit:
                # 使用selectbox选择文档
                doc_options = [f"ID: {d.get('id', i)} | {d.get('title', '')}" for i, d in enumerate(docs_for_edit)]
                selected_idx = st.selectbox(
                    "选择文档",
                    range(len(docs_for_edit)),
                    format_func=lambda x: doc_options[x],
                    key="edit_doc_select"
                )
                current = docs_for_edit[selected_idx]

                # 使用form编辑文档
                with st.form("edit_doc_form"):
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        edit_title = st.text_input("标题", value=str(current.get('title', '')))
                    with col_e2:
                        edit_category = st.text_input("分类", value=str(current.get('category', '未分类')))

                    edit_text = st.text_area("内容", value=str(current.get('text', '')), height=120)

                    # 违规类型多选
                    # 处理已有数据：将逗号分隔的字符串转换为列表（支持中文和英文逗号）
                    current_violation = current.get('violation_type', '无')
                    if isinstance(current_violation, str):
                        # 先替换中文逗号为英文逗号，再分割
                        current_violation_normalized = current_violation.replace('，', ',').replace(';', ',').replace('；', ',')
                        current_violation_list = [v.strip() for v in current_violation_normalized.split(',') if v.strip()]
                        # 过滤掉不在选项中的值
                        current_violation_list = [v for v in current_violation_list if v in Config.VIOLATION_TYPES]
                        if not current_violation_list:
                            current_violation_list = ["无"]
                    else:
                        current_violation_list = ["无"]

                    edit_violation_type_list = st.multiselect(
                        "违规类型（可多选）",
                        Config.VIOLATION_TYPES,
                        default=current_violation_list,
                        help="选择一种或多种违规类型"
                    )
                    # 将列表转换为逗号分隔的字符串存储
                    edit_violation_type_str = ", ".join(edit_violation_type_list) if edit_violation_type_list else "无"

                    col_e3, col_e4 = st.columns(2)
                    with col_e3:
                        st.markdown("")  # 占位，保持布局
                    with col_e4:
                        severity_options = ["无", "低", "中", "高"]
                        default_sev = current.get('severity', '无')
                        sev_idx = severity_options.index(default_sev) if default_sev in severity_options else 0
                        edit_severity = st.selectbox("严重程度", severity_options, index=sev_idx)

                    col_e5, col_e6 = st.columns(2)
                    with col_e5:
                        edit_suggestion = st.text_input("处理建议", value=str(current.get('suggestion', '无')))
                    with col_e6:
                        edit_sources = st.text_input("来源", value=str(current.get('sources', '')))

                    submit_btn = st.form_submit_button("✏️ 保存编辑", type="primary")

                if submit_btn:
                    try:
                        st.session_state.vector_db.update_document(
                            selected_idx,
                            text=edit_text,
                            category=edit_category,
                            title=edit_title,
                            violation_type=edit_violation_type_str,
                            severity=edit_severity,
                            suggestion=edit_suggestion,
                            sources=edit_sources
                        )
                        st.success("✅ 已保存（并同步写回CSV）")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
            else:
                st.info("💡 知识库为空，无法编辑")

            st.subheader("删除知识")
            docs_for_del = st.session_state.vector_db.get_all_documents()
            if docs_for_del:
                # 使用selectbox选择要删除的文档
                del_options = [f"ID: {d.get('id', i)} | {d.get('title', '')}" for i, d in enumerate(docs_for_del)]
                del_idx = st.selectbox(
                    "选择要删除的文档",
                    range(len(docs_for_del)),
                    format_func=lambda x: del_options[x],
                    key="del_doc_select"
                )
                if st.button("🗑️ 删除", use_container_width=True):
                    try:
                        st.session_state.vector_db.delete_document(del_idx)
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



    # 初始化审核模式
    if 'audit_mode' not in st.session_state:
        st.session_state.audit_mode = True  # 默认使用审核模式

    col_mode, col_rerank, col_show = st.columns([1, 1, 1])
    with col_mode:
        mode = st.radio("功能模式", ["内容审核", "智能问答"], horizontal=True)
        st.session_state.audit_mode = (mode == "内容审核")
    with col_rerank:
        use_rerank = st.checkbox("启用重排序", value=True)
    with col_show:
        show_prompt = st.checkbox("显示提示词", value=False)

    if not st.session_state.llm_configured:
        st.warning("⚠️ 请先在侧边栏配置大模型API")

    if st.session_state.audit_mode:
        # 内容审核模式
        query = st.text_area(
            "待审核文本：",
            height=150,
            placeholder="请输入需要审核的文本内容，例如：某用户准备发布的一篇文章、一条评论等..."
        )

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            rerank_top_k = st.number_input("检索重排数量", min_value=1, max_value=5, value=3, step=1)
        with col_k2:
            audit_temperature = st.slider("审核温度", 0.0, 1.0, 0.3, 0.1, help="较低温度可使审核结果更稳定")

        if st.button("🔍 开始审核", type="primary", use_container_width=True):
            if query:
                if not st.session_state.llm_configured:
                    st.error("❌ 请先配置大模型API")
                else:
                    with st.spinner("🔄 正在审核中，请稍候..."):
                        result = st.session_state.rag_system.audit_content(
                            query,
                            use_rerank,
                            temperature=audit_temperature,
                            rerank_top_k=rerank_top_k
                        )

                        # 展示审核结果
                        audit_result = result.get('audit_result', {})

                        # 根据审核结果显示不同的颜色
                        result_text = audit_result.get('result', '需人工复核')
                        if result_text == '合规':
                            st.success(f"✅ 审核结果：{result_text}")
                        elif result_text == '不合规':
                            st.error(f"❌ 审核结果：{result_text}")
                        elif result_text == '可能不合规':
                            st.warning(f"⚠️ 审核结果：{result_text}")
                        else:
                            st.info(f"ℹ️ 审核结果：{result_text}")

                        # 显示详细信息
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("违规类型", audit_result.get('violation_type', '无'))
                            st.metric("严重程度", audit_result.get('severity', '无'))
                        with col_res2:
                            st.metric("参考来源", audit_result.get('sources', '无'))

                        st.subheader("📋 审核详情")
                        st.markdown(f"**原因：**\n{audit_result.get('reason', '无')}")
                        st.markdown(f"**处理建议：**\n{audit_result.get('suggestion', '无')}")

                        # 显示完整LLM回答
                        with st.expander("📝 查看完整审核报告"):
                            st.markdown(result['answer'])

                        if show_prompt:
                            with st.expander("📋 查看审核提示词"):
                                st.code(result['prompt'], language="text")

                        # 显示检索到的知识片段
                        with st.expander("🔎 查看检索到的知识片段"):
                            st.markdown("**参考知识库内容：**")
                            for i, doc in enumerate(result['retrieved_docs'], 1):
                                score = doc.get('rerank_score', doc.get('score', 0))
                                st.markdown(f"""
                                ---
                                **片段 {i}** | 相关度: `{score:.3f}`
                                - **分类**: {doc.get('category', '未分类')}
                                - **标题**: {doc.get('title', '')}
                                - **违规类型**: {doc.get('violation_type', '无')}
                                - **严重程度**: {doc.get('severity', '无')}
                                - **内容**: {doc.get('text', '')[:300]}{'...' if len(doc.get('text', '')) > 300 else ''}
                                - **处理建议**: {doc.get('suggestion', '无')}
                                - **来源**: {doc.get('sources', '无')}
                                """)
            else:
                st.warning("⚠️ 请输入待审核文本")

    else:
        # 智能问答模式
        query = st.text_area("请输入您的问题：", height=120,
                             placeholder="例如：关于某历史事件的准确描述是什么？")

        col_k, col_t = st.columns(2)
        with col_k:
            rerank_top_k = st.number_input("重排数量", min_value=0, max_value=5, value=int(Config.RERANK_TOP_K), step=1)
        with col_t:
            qa_temperature = st.slider("温度参数", 0.0, 1.0, 0.7, 0.1)

        if st.button("🔍 提交查询", type="primary", use_container_width=True):
            if query:
                if not st.session_state.llm_configured:
                    st.error("❌ 请先配置大模型API")
                else:
                    with st.spinner("🤔 正在思考..."):
                        result = st.session_state.rag_system.answer(
                            query,
                            use_rerank,
                            temperature=qa_temperature,
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
                                score = doc.get('rerank_score', doc.get('score', 0))
                                st.markdown(f"""
                                ---
                                **片段 {i}** | 相关度: `{score:.3f}`
                                - **类别**: {doc.get('category', '未分类')}
                                - **内容**: {doc.get('text', '')[:200]}{'...' if len(doc.get('text', '')) > 200 else ''}
                                """)
            else:
                st.warning("⚠️ 请输入问题")

    st.markdown("---")
    st.header("📚 知识库浏览")

    docs = st.session_state.vector_db.get_all_documents()

    if docs:
        search_term = st.text_input("🔍 搜索知识库")

        filtered_docs = docs
        if search_term:
            filtered_docs = [d for d in docs if search_term.lower() in str(d.get('text', '')).lower()]

        st.write(f"显示 {len(filtered_docs)} / {len(docs)} 条")

        for doc in filtered_docs[:20]:
            with st.expander(f"ID: {doc.get('id', '')} | {doc.get('category', '未分类')} | {doc.get('title', '')}", expanded=False):
                st.write(f"**标题**: {doc.get('title', '')}")
                st.write(f"**内容**: {doc.get('text', '')}")
                st.write(f"**违规类型**: {doc.get('violation_type', '无')}")
                st.write(f"**严重程度**: {doc.get('severity', '无')}")
                st.write(f"**处理建议**: {doc.get('suggestion', '无')}")
                st.write(f"**来源**: {doc.get('sources', '')}")
                st.caption(f"时间: {doc.get('timestamp', '')}")
    else:
        st.info("💡 知识库为空，请先导入数据")


if __name__ == "__main__":
    main()
