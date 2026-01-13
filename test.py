"""
涉政内容审核RAG系统
功能：向量数据库构建、管理、检索增强生成、交互界面
"""

import os
import csv
import json
import numpy as np
from typing import List, Dict, Tuple
import pickle
from datetime import datetime

# 向量化和相似度计算
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM集成（使用OpenAI API格式）
import anthropic


class VectorDatabase:
    """向量数据库管理类"""

    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        """初始化向量数据库"""
        self.embedding_model = SentenceTransformer(model_name)
        self.documents = []  # 存储原始文档
        self.embeddings = None  # 存储向量
        self.metadata = []  # 存储元数据

    def load_from_csv(self, csv_file: str, text_column: str, metadata_columns: List[str] = None):
        """从CSV文件加载知识库"""
        print(f"正在从 {csv_file} 加载数据...")

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get(text_column, '')
                if text:
                    self.documents.append(text)

                    # 提取元数据
                    meta = {'id': len(self.documents) - 1}
                    if metadata_columns:
                        for col in metadata_columns:
                            meta[col] = row.get(col, '')
                    self.metadata.append(meta)

        # 生成向量
        print(f"正在生成 {len(self.documents)} 个文档的向量...")
        self.embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        print(f"向量数据库构建完成！共 {len(self.documents)} 条记录")

    def add_document(self, text: str, metadata: Dict = None):
        """添加单个文档"""
        self.documents.append(text)

        meta = {'id': len(self.documents) - 1}
        if metadata:
            meta.update(metadata)
        self.metadata.append(meta)

        # 生成新文档的向量
        new_embedding = self.embedding_model.encode([text])

        if self.embeddings is None:
            self.embeddings = new_embedding
        else:
            self.embeddings = np.vstack([self.embeddings, new_embedding])

        print(f"文档添加成功，ID: {meta['id']}")
        return meta['id']

    def delete_document(self, doc_id: int):
        """删除文档"""
        if 0 <= doc_id < len(self.documents):
            self.documents.pop(doc_id)
            self.metadata.pop(doc_id)
            self.embeddings = np.delete(self.embeddings, doc_id, axis=0)

            # 更新ID
            for i in range(doc_id, len(self.metadata)):
                self.metadata[i]['id'] = i

            print(f"文档 {doc_id} 删除成功")
            return True
        else:
            print(f"文档 {doc_id} 不存在")
            return False

    def update_document(self, doc_id: int, new_text: str, new_metadata: Dict = None):
        """更新文档"""
        if 0 <= doc_id < len(self.documents):
            self.documents[doc_id] = new_text

            if new_metadata:
                self.metadata[doc_id].update(new_metadata)

            # 重新生成向量
            new_embedding = self.embedding_model.encode([new_text])
            self.embeddings[doc_id] = new_embedding[0]

            print(f"文档 {doc_id} 更新成功")
            return True
        else:
            print(f"文档 {doc_id} 不存在")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, Dict]]:
        """向量检索"""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        # 查询向量化
        query_embedding = self.embedding_model.encode([query])

        # 计算相似度
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.documents[idx],
                self.metadata[idx]
            ))

        return results

    def save(self, filepath: str):
        """保存向量数据库"""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"向量数据库已保存到 {filepath}")

    def load(self, filepath: str):
        """加载向量数据库"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.documents = data['documents']
        self.embeddings = data['embeddings']
        self.metadata = data['metadata']
        print(f"向量数据库已从 {filepath} 加载，共 {len(self.documents)} 条记录")


class RAGSystem:
    """RAG系统核心类"""

    def __init__(self, vector_db: VectorDatabase, api_key: str = None):
        """初始化RAG系统"""
        self.vector_db = vector_db
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            print("警告：未提供API密钥，将使用模拟模式")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """召回阶段：从向量数据库检索相关文档"""
        results = self.vector_db.search(query, top_k=top_k)

        retrieved_docs = []
        for doc_id, score, text, metadata in results:
            retrieved_docs.append({
                'id': doc_id,
                'score': score,
                'text': text,
                'metadata': metadata
            })

        return retrieved_docs

    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """重排阶段：基于相关性分数重新排序"""
        # 简单重排：按分数排序并取top-k
        sorted_docs = sorted(documents, key=lambda x: x['score'], reverse=True)
        return sorted_docs[:top_k]

    def generate_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """生成带有上下文的提示词"""
        context = "\n\n".join([
            f"[参考资料 {i + 1}]\n{doc['text']}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""你是一个专业的涉政内容审核助手。请根据以下参考资料回答用户的问题。

参考资料：
{context}

用户问题：{query}

请基于上述参考资料，准确、客观地回答问题。如果参考资料中没有相关信息，请明确说明。注意保持政治立场正确，事实准确。"""

        return prompt

    def generate_response(self, query: str, top_k_retrieve: int = 5, top_k_rerank: int = 3) -> Dict:
        """生成回答（完整RAG流程）"""
        # 1. 召回
        print(f"\n正在检索相关文档...")
        retrieved_docs = self.retrieve(query, top_k=top_k_retrieve)
        print(f"召回 {len(retrieved_docs)} 个相关文档")

        # 2. 重排
        print(f"正在重排序...")
        reranked_docs = self.rerank(query, retrieved_docs, top_k=top_k_rerank)
        print(f"重排后保留 {len(reranked_docs)} 个文档")

        # 3. 生成提示词
        prompt = self.generate_prompt(query, reranked_docs)

        # 4. 调用LLM生成回答
        print(f"正在生成回答...")

        if self.client:
            try:
                message = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = message.content[0].text
            except Exception as e:
                response_text = f"API调用失败: {str(e)}"
        else:
            # 模拟模式
            response_text = f"[模拟回答] 基于检索到的 {len(reranked_docs)} 个参考资料，针对您的问题：{query}\n\n这是一个模拟回答。请配置ANTHROPIC_API_KEY以使用真实的AI生成。"

        return {
            'query': query,
            'response': response_text,
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


class InteractiveInterface:
    """交互界面类"""

    def __init__(self, rag_system: RAGSystem):
        """初始化交互界面"""
        self.rag_system = rag_system
        self.history = []

    def display_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 60)
        print("涉政内容审核RAG系统")
        print("=" * 60)
        print("1. 查询问题（RAG检索+生成）")
        print("2. 管理向量数据库")
        print("3. 查看历史记录")
        print("4. 保存向量数据库")
        print("5. 加载向量数据库")
        print("0. 退出系统")
        print("=" * 60)

    def display_db_menu(self):
        """显示数据库管理菜单"""
        print("\n" + "-" * 60)
        print("向量数据库管理")
        print("-" * 60)
        print("1. 添加文档")
        print("2. 删除文档")
        print("3. 更新文档")
        print("4. 查询文档")
        print("5. 查看数据库统计")
        print("0. 返回主菜单")
        print("-" * 60)

    def query_interface(self):
        """查询界面"""
        query = input("\n请输入您的问题: ").strip()
        if not query:
            print("问题不能为空！")
            return

        print("\n正在处理您的问题...")
        result = self.rag_system.generate_response(query)

        # 显示结果
        print("\n" + "=" * 60)
        print("检索到的参考资料：")
        print("=" * 60)
        for i, doc in enumerate(result['reranked_docs'], 1):
            print(f"\n[资料 {i}] (相似度: {doc['score']:.4f})")
            print(doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'])

        print("\n" + "=" * 60)
        print("AI生成的回答：")
        print("=" * 60)
        print(result['response'])
        print("=" * 60)

        self.history.append(result)

    def manage_database(self):
        """数据库管理界面"""
        while True:
            self.display_db_menu()
            choice = input("\n请选择操作: ").strip()

            if choice == '1':
                text = input("请输入文档内容: ").strip()
                if text:
                    self.rag_system.vector_db.add_document(text)

            elif choice == '2':
                try:
                    doc_id = int(input("请输入要删除的文档ID: ").strip())
                    self.rag_system.vector_db.delete_document(doc_id)
                except ValueError:
                    print("请输入有效的数字ID")

            elif choice == '3':
                try:
                    doc_id = int(input("请输入要更新的文档ID: ").strip())
                    new_text = input("请输入新的文档内容: ").strip()
                    if new_text:
                        self.rag_system.vector_db.update_document(doc_id, new_text)
                except ValueError:
                    print("请输入有效的数字ID")

            elif choice == '4':
                query = input("请输入查询内容: ").strip()
                if query:
                    results = self.rag_system.vector_db.search(query, top_k=5)
                    print(f"\n找到 {len(results)} 个相关文档：")
                    for doc_id, score, text, metadata in results:
                        print(f"\nID: {doc_id} | 相似度: {score:.4f}")
                        print(text[:150] + "..." if len(text) > 150 else text)

            elif choice == '5':
                print(f"\n数据库统计信息：")
                print(f"文档总数: {len(self.rag_system.vector_db.documents)}")
                print(
                    f"向量维度: {self.rag_system.vector_db.embeddings.shape[1] if self.rag_system.vector_db.embeddings is not None else 0}")

            elif choice == '0':
                break

            else:
                print("无效的选择，请重新输入")

    def view_history(self):
        """查看历史记录"""
        if not self.history:
            print("\n暂无历史记录")
            return

        print(f"\n共有 {len(self.history)} 条历史记录：")
        for i, record in enumerate(self.history, 1):
            print(f"\n[{i}] {record['timestamp']}")
            print(f"问题: {record['query']}")
            print(f"回答: {record['response'][:100]}...")

    def run(self):
        """运行交互界面"""
        print("\n欢迎使用涉政内容审核RAG系统！")

        while True:
            self.display_menu()
            choice = input("\n请选择操作: ").strip()

            if choice == '1':
                self.query_interface()

            elif choice == '2':
                self.manage_database()

            elif choice == '3':
                self.view_history()

            elif choice == '4':
                filepath = input("请输入保存路径 (默认: vector_db.pkl): ").strip()
                filepath = filepath or "vector_db.pkl"
                self.rag_system.vector_db.save(filepath)

            elif choice == '5':
                filepath = input("请输入加载路径: ").strip()
                if filepath and os.path.exists(filepath):
                    self.rag_system.vector_db.load(filepath)
                else:
                    print("文件不存在！")

            elif choice == '0':
                print("\n感谢使用！再见！")
                break

            else:
                print("无效的选择，请重新输入")


def create_sample_csv():
    """创建示例CSV文件"""
    sample_data = [
        {
            'id': '1',
            'category': '历史事件',
            'content': '中华人民共和国于1949年10月1日成立，标志着中国人民站起来了，开启了中华民族伟大复兴的历史新纪元。'
        },
        {
            'id': '2',
            'category': '政策法规',
            'content': '《中华人民共和国宪法》是国家的根本大法，具有最高的法律效力。宪法规定了国家的根本制度和根本任务。'
        },
        {
            'id': '3',
            'category': '重大会议',
            'content': '党的二十大是在全党全国各族人民迈上全面建设社会主义现代化国家新征程、向第二个百年奋斗目标进军的关键时刻召开的一次十分重要的大会。'
        },
        {
            'id': '4',
            'category': '法律法规',
            'content': '网络安全法规定，国家保护公民、法人和其他组织的个人信息安全，任何个人和组织不得窃取或者以其他非法方式获取个人信息。'
        },
        {
            'id': '5',
            'category': '历史事件',
            'content': '改革开放是1978年12月十一届三中全会开始实行的对内改革、对外开放的政策，是中国共产党历史上具有深远意义的伟大转折。'
        }
    ]

    with open('political_knowledge_base.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'content'])
        writer.writeheader()
        writer.writerows(sample_data)

    print("示例知识库文件 'political_knowledge_base.csv' 创建成功！")


def main():
    """主函数"""
    print("涉政内容审核RAG系统初始化中...")

    # 检查是否存在示例CSV
    if not os.path.exists('political_knowledge_base.csv'):
        print("\n未找到知识库文件，正在创建示例文件...")
        create_sample_csv()

    # 初始化向量数据库
    vector_db = VectorDatabase()

    # 加载CSV知识库
    csv_file = input("\n请输入CSV文件路径 (默认: political_knowledge_base.csv): ").strip()
    csv_file = csv_file or 'political_knowledge_base.csv'

    if os.path.exists(csv_file):
        text_column = input("请输入文本内容所在列名 (默认: content): ").strip() or 'content'
        metadata_cols = input("请输入元数据列名，用逗号分隔 (默认: id,category): ").strip()
        metadata_cols = metadata_cols.split(',') if metadata_cols else ['id', 'category']

        vector_db.load_from_csv(csv_file, text_column, metadata_cols)
    else:
        print(f"文件 {csv_file} 不存在！")
        return

    # 初始化RAG系统
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("\n提示：未检测到ANTHROPIC_API_KEY环境变量")
        api_key = input("请输入您的Anthropic API Key (留空使用模拟模式): ").strip()

    rag_system = RAGSystem(vector_db, api_key)

    # 启动交互界面
    interface = InteractiveInterface(rag_system)
    interface.run()


if __name__ == "__main__":
    main()