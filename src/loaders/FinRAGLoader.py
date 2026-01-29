import os
import json
import sys
import torch
import numpy as np
import faiss
import re
import base64
import time
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from PIL import Image
import uuid
import re

# 引入评估所需的库
try:
    from rouge_score import rouge_scorer
except ImportError:
    # 即使只用Model Eval，保留此检查以防依赖报错，但在本逻辑中不会使用它
    rouge_scorer = None

# Adjust path to ensure we can import from src and scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.loaders.base_loader import BaseDataLoader, StandardSample, PageElement
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
from scripts.qwen3_vl_reranker_client import Qwen3VLReranker
from src.agents.ElementExtractor import ElementExtractor
from src.utils.llm_helper import create_llm_caller

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def encode_image_to_base64(image_path):
    """Convert image to base64 encoding."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_pdf_prefix(filename):
    """
    从图片文件名中提取PDF原始前缀，移除页码后缀（如 _116.png-2）
    
    Args:
        filename (str): 待处理的文件名字符串
        
    Returns:
        str: 提取后的前缀。如果匹配失败，则返回原字符串。
    """
    # 逻辑：匹配从开头直到最后一个“_数字.png”之前的内容
    # ^       : 匹配开头
    # (.*)    : 捕获组，匹配前缀
    # (?=     : 正向预查（不计入结果）
    # _\d+    : 下划线后跟一个或多个数字（页码）
    # \.png   : 匹配 .png 扩展名
    pattern = r"^(.*)(?=_\d+\.png)"
    
    match = re.search(pattern, filename)
    return match.group(1) if match else filename

class FinRAGLoader(BaseDataLoader):
    def __init__(self, data_root: str, lang: str = "ch", embedding_model=None, rerank_model=None, extractor: Optional[ElementExtractor] = None):
        super().__init__(data_root)
        self.lang = lang.lower()
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.extractor = extractor
        
        # --- 路径配置 ---
        self.query_path = os.path.join(data_root, "data", "queries", f"queries_{self.lang}.json")
        self.corpus_root = os.path.join(data_root, "data", "corpus", self.lang, "img")
        self.qrels_path = os.path.join(data_root, "data", "qrels", f"qrels_{self.lang}.tsv")
        self.citation_root = os.path.join(data_root, "data", "citation_labels", "citation_labels_new")
        
        # 索引路径
        cache_dir = os.path.join(data_root, "data", "indices")
        os.makedirs(cache_dir, exist_ok=True)
        self.doc_map_path = os.path.join(cache_dir, f"finrag_{self.lang}_hnsw_docmap.json")
        
        self.doc_id_map = {} 
        self.llm_caller = None
        self.llm_element_judge = False

    def _load_qrels(self) -> Dict[str, List[str]]:
        """读取 qrels TSV 文件。"""
        qrels_map = {}
        if not os.path.exists(self.qrels_path):
            print(f"Warning: Qrels file not found at {self.qrels_path}. GT IDs will be empty.")
            return qrels_map
            
        print(f"Loading qrels from: {self.qrels_path}")
        with open(self.qrels_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("query-id"):
                    continue
                parts = line.split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    cid = parts[1]
                    if qid not in qrels_map:
                        qrels_map[qid] = []
                    qrels_map[qid].append(cid)
        return qrels_map

    def load_bbox_data(self) -> None:
        """
        加载 selected_200_with_bboxes.json 数据集。
        """
        json_file_path = os.path.join(self.citation_root, "selected_200_with_bboxes.json")
        img_root_dir = self.citation_root

        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Selected dataset file not found: {json_file_path}")
        
        print(f"Loading selected data from: {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            qid = str(item.get("query-id") or item.get("id", str(count)))
            query_text = item.get("query", "")
            gold_answer = item.get("answer", "")
            
            # 解析 Metadata
            extra_info = {
                'category': item.get('category'),
                'answer_type': item.get('answer_type'),
                'human_eval': item.get('human_eval'),
                'from_pages': item.get('from_pages')
            }

            # 解析 Ground Truth Pages 和 BBoxes
            gold_pages = []
            gold_elements = []
            
            img_paths_map = item.get("img_paths", {})
            bboxes_map = item.get("bboxes", {})
            
            for page_id, rel_path in img_paths_map.items():
                full_img_path = os.path.normpath(os.path.join(img_root_dir, rel_path))
                gold_pages.append(full_img_path)
                
                page_bboxes = bboxes_map.get(page_id, [])
                for box in page_bboxes:
                    x1 = int(box.get("xmin", 0) * 1000)
                    y1 = int(box.get("ymin", 0) * 1000)
                    x2 = int(box.get("xmax", 0) * 1000)
                    y2 = int(box.get("ymax", 0) * 1000)
                    
                    pe = PageElement(
                        bbox=[x1, y1, x2, y2],
                        type="evidence", 
                        content=gold_answer,
                        corpus_id=full_img_path,
                        crop_path=full_img_path
                    )
                    gold_elements.append(pe)

            sample = StandardSample(
                qid=qid, 
                query=query_text, 
                dataset=f"finrag-{self.lang}",
                data_source=gold_pages[0],
                gold_answer=gold_answer,
                gold_elements=gold_elements,
                gold_pages=gold_pages, 
                extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
            
        print(f"✅ Successfully loaded {count} samples from selected_200_with_bboxes.json.")

    def load_data(self) -> None:
        
        if self.lang == 'bbox':
            self.load_bbox_data()
            return
        
        """加载 Query 数据集并关联 Qrels。"""
        if not os.path.exists(self.query_path):
            raise FileNotFoundError(f"Query file not found: {self.query_path}")
        
        qrels_map = self._load_qrels()
        print(f"Loading queries from: {self.query_path}")
        with open(self.query_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        count = 0
        for item in data:
            query_text = item.get("query", "") or item.get("question", "")
            if not query_text:
                continue
            qid = str(item.get("id") or item.get("query-id") or item.get("_id") or "")
            gold_answer = item.get("answer", "") or item.get("response", "")
            extra_info = {'category': item.get('category', None), 'answer_type': item.get('answer_type', None), 'from_pages': item.get('from_pages', None)}
            gold_pages = qrels_map.get(qid, [])
            
            sample = StandardSample(
                qid=qid, query=query_text, dataset=f"finrag-{self.lang}",
                data_source=extract_pdf_prefix(qid), gold_answer=gold_answer,
                gold_elements=None, gold_pages=gold_pages, extra_info=extra_info
            )
            self.samples.append(sample)
            count += 1
        print(f"✅ Successfully loaded {count} queries.")
    

    def _get_all_image_paths(self) -> List[str]:
        print(f"Scanning images in {self.corpus_root}...")
        image_files = []
        for root, dirs, files in os.walk(self.corpus_root):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        image_files.sort()
        print(f"Found {len(image_files)} images.")
        return image_files

    def _embed_images(self, image_paths: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"image": p} for p in image_paths]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def _embed_text(self, text: str) -> np.ndarray:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized.")
        inputs = [{"text": text, "instruction": "Represent the user's input."}]
        embeddings = self.embedding_model.process(inputs)
        return embeddings.cpu().numpy().astype('float32')

    def _pdf_to_images(self, pdf_path: str) -> Dict[int, str]:
        """
        修改后逻辑：
        根据 data_source (PDF文件名/前缀) 在 corpus_root 中查找已存在的页面图片。
        图片命名格式预期为: {pdf_prefix}_{page_num}.png
        例如: 1-s2.0-S0304405X24001697-main_17.png
        """
        import glob
        
        # 1. 提取 PDF 前缀 (移除路径和 .pdf 后缀)
        # pdf_path 可能是 "xxx.pdf" 也可能是单纯的 ID "xxx"
        pdf_prefix = os.path.basename(pdf_path)
        if pdf_prefix.lower().endswith('.pdf'):
            pdf_prefix = pdf_prefix[:-4]

        image_map = {}
        
        # 2. 检查 corpus 目录是否存在
        if not os.path.exists(self.corpus_root):
            print(f"Warning: Corpus root directory not found: {self.corpus_root}")
            return {}

        # 3. 构建搜索模式: corpus_root/前缀_*.png
        # 注意：这里加 "_" 是为了防止前缀部分匹配 (如 doc_1 匹配到 doc_10_1.png)
        search_pattern = os.path.join(self.corpus_root, f"{pdf_prefix}_*.png")
        
        # 4. 查找匹配的文件
        found_files = glob.glob(search_pattern)
        
        for file_path in found_files:
            filename = os.path.basename(file_path)
            
            # 5. 使用正则提取末尾的页码
            # 匹配逻辑: 查找文件名末尾的 "_数字.png"
            match = re.search(r"_(\d+)\.png$", filename)
            if match:
                try:
                    # 提取页码并存入字典
                    page_num = int(match.group(1))
                    image_map[page_num] = file_path
                except ValueError:
                    continue
        
        if not image_map:
            # 调试信息：如果没有找到图片，可能是前缀不匹配或者目录不对
            print(f"Warning: No pre-processed images found for prefix '{pdf_prefix}' in {self.corpus_root}")

        return image_map

    def pipeline(self, query: str, image_paths: List[str] = None, top_k: int = 10) -> List[PageElement]:
        """
        修改后的 Pipeline：直接处理 PDF 文档而非检索池
        """
        if not image_paths:
            return []

        # 1. 解析 PDF 或直接使用图片路径
        all_pages_to_process = []
        for path in image_paths:
            if not path.lower().endswith('.png'):
                page_map = self._pdf_to_images(path)
                # 将 PDF 每一页包装成待排序的 PageElement
                for p_idx in sorted(page_map.keys()):
                    all_pages_to_process.append(PageElement(
                        bbox=[0, 0, 1000, 1000],
                        type="page_image",
                        corpus_id=os.path.basename(page_map[p_idx]),
                        corpus_path=page_map[p_idx],
                        crop_path=page_map[p_idx]
                    ))
            else:
                all_pages_to_process.append(PageElement(
                    bbox=[0, 0, 1000, 1000],
                    type="page_image",
                    corpus_path=path,
                    crop_path=path
                ))

        # 2. 如果页面较多，使用 Reranker 进行筛选
        if self.rerank_model and len(all_pages_to_process) > top_k:
            ranked_pages = self.rerank(query, all_pages_to_process)
            target_pages = ranked_pages[:top_k]
            target_pages = [ page for page in target_pages if page.retrieval_score >= 0.1]
        else:
            target_pages = all_pages_to_process[:top_k]

        # 3. 调用 ElementExtractor 进行细粒度提取
        elements = self.extract_elements_from_pages(target_pages, query)
        return elements[:top_k]


    def rerank(self, query: str, pages: List[PageElement]) -> List[PageElement]:
        """Step 2: Reranking using Qwen3-VL-Reranker"""
        if not self.rerank_model or not pages:
            return pages
        print(f"Reranking {len(pages)} pages...")
        
        documents_input = [{"image": page.corpus_path} for page in pages]
        rerank_input = {
            "instruction": "Given a search query, retrieve relevant candidates that answer the query.",
            "query": {"text": query},
            "documents": documents_input,
            "fps": 1.0 
        }
        
        scores = self.rerank_model.process(rerank_input)
        if len(scores) != len(pages):
            print(f"Warning: Reranker returned {len(scores)} scores for {len(pages)} pages.")
            return pages

        for page, score in zip(pages, scores):
            page.retrieval_score = score
            
        sorted_pages = sorted(pages, key=lambda x: x.retrieval_score, reverse=True)
        return sorted_pages

    def extract_elements_from_pages(self, pages: List[PageElement], query: str) -> List[PageElement]:
        """Step 3: Downstream Element Extraction using ElementExtractor."""
        if self.extractor is None:
            print("Warning: ElementExtractor is not initialized, skipping fine-grained extraction.")
            return pages 

        workspace_dir = os.path.abspath(os.path.join(os.getcwd(), "workspace", "crops"))
        os.makedirs(workspace_dir, exist_ok=True)

        fine_grained_elements = []
        
        for page in tqdm(pages, desc="Extracting Elements"):
            image_path = page.crop_path
            
            if not image_path or not os.path.exists(image_path):
                print(f"Warning: Image path not found: {image_path}")
                continue

            try:
                agent_output = self.extractor.run_agent(
                    user_text=query,
                    image_paths=[image_path]
                )
                
                if not agent_output:
                    continue

                predictions = agent_output.get("predictions", [])
                if not predictions:
                    continue
                
                last_msg_content = predictions[-1].get("content", "")
                
                json_str = "[]"
                match = re.search(r'```json(.*?)```', last_msg_content, re.DOTALL)
                if match:
                    json_str = match.group(1).strip()
                else:
                    start = last_msg_content.find('[')
                    end = last_msg_content.rfind(']')
                    if start != -1 and end != -1:
                        json_str = last_msg_content[start:end+1]

                try:
                    extracted_data = json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"JSON Decode Error for page {page.corpus_id}")
                    extracted_data = []

                if isinstance(extracted_data, list):
                    try:
                        original_img = Image.open(image_path)
                        img_w, img_h = original_img.size
                    except Exception as e:
                        print(f"Failed to open image {image_path}: {e}")
                        continue

                    for item in extracted_data:
                        bbox = item.get("bbox", [0, 0, 0, 0])
                        evidence = item.get("evidence", "")
                        
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            x1 = int(x1 / 1000 * img_w)
                            y1 = int(y1 / 1000 * img_h)
                            x2 = int(x2 / 1000 * img_w)
                            y2 = int(y2 / 1000 * img_h)
                            
                            x1 = max(0, x1); y1 = max(0, y1); x2 = min(img_w, x2); y2 = min(img_h, y2)
                            
                            current_crop_path = image_path
                            
                            if x2 > x1 and y2 > y1:
                                try:
                                    cropped_img = original_img.crop((x1, y1, x2, y2))
                                    filename = f"{os.path.basename(page.corpus_id).split('.')[0]}_{uuid.uuid4().hex[:8]}.jpg"
                                    save_path = os.path.join(workspace_dir, filename)
                                    cropped_img.save(save_path)
                                    current_crop_path = save_path
                                except Exception as crop_err:
                                    print(f"Error cropping image: {crop_err}")
                        else:
                            current_crop_path = image_path

                        element = PageElement(
                            bbox=bbox, 
                            type="evidence",
                            content=evidence,
                            corpus_id=page.corpus_id,
                            corpus_path=page.corpus_path,
                            crop_path=current_crop_path
                        )
                        
                        if hasattr(page, 'retrieval_score'):
                            element.retrieval_score = page.retrieval_score
                            
                        fine_grained_elements.append(element)
                        
            except Exception as e:
                print(f"Error during extraction on {page.corpus_id}: {e}")

        if not fine_grained_elements:
            print("No fine-grained elements extracted.")
            
        return fine_grained_elements

    # --------------------------------------------------------------------------------
    # Evaluation Methods (Modified to ONLY use Model Eval)
    # --------------------------------------------------------------------------------

    def _evaluate_answer_correctness(self, sample: StandardSample) -> Dict[str, Any]:
        """
        MODIFIED: 强制对所有类型的样本使用 LLM 进行正确性评估 (Model Eval)。
        忽略 answer_type (short/long) 的区别。
        """
        query_text = sample.query
        expected_answer = sample.gold_answer
        actual_answer = sample.extra_info.get('final_answer', "error output") if sample.extra_info else "error output"

        model_eval = 0

        # 如果没有 LLM Caller 或答案为空，直接判定为 0
        if "error output" in actual_answer or not self.llm_caller:
            model_eval = 0
        else:
            # 统一构建 Prompt，不区分长短答案
            evaluation_prompt = (
                f"Question: {query_text}\n"
                f"Ground_truth: {expected_answer}\n"
                f"Model_answer: {actual_answer}\n"
                f"Is the model answer correct? You only need to output 'true' for correct or 'false' for incorrect. "
                f"If the model answer does not contain any information, it should be judged as 'false'."
            )
            try:
                response_core = self.llm_caller(evaluation_prompt)
                model_eval = 1 if "true" in response_core.lower() else 0
            except Exception as e:
                print(f"Error during model eval for QID {sample.qid}: {e}")
                model_eval = 0

        return {
            "model_eval": model_eval
        }

    def _check_images_entailment(self, elements: List[PageElement], answer: str) -> bool:
        """
        检查图片证据是否蕴含答案 (保持原逻辑，这本身就是一种 Model Eval)。
        """
        if not elements or not self.llm_caller:
            return False

        usr_msg = [
            {"type": "text", "text": f"Answer: {answer}"},
            {"type": "text", "text": "Please judge whether these pages cover the answer, your answer can only be 'yes' or 'no'. Only generate one response for each input group, do not output any explanation."},
            {"type": "text", "text": "Here is my file page:"}
        ]

        valid_images = 0
        for el in elements:
            if el.crop_path and os.path.exists(el.crop_path):
                encoded_image = encode_image_to_base64(el.crop_path)
                if encoded_image:
                    usr_msg.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}", "detail": "auto"}
                    })
                    valid_images += 1
        
        if valid_images == 0:
            return False

        attempt, max_retries, retry_delay = 0, 3, 5
        while attempt < max_retries:
            try:
                response_core = self.llm_caller(usr_msg)
                return "yes" in response_core.lower()
            except Exception as e:
                attempt += 1
                print(f"Error calling model for entailment: {e}. Retry {attempt}/{max_retries}.")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    return False
        return False

    # --------------------------------------------------------------------------------
    # 增加的评估辅助函数
    # --------------------------------------------------------------------------------

    def _compute_element_metrics(self, pred_elements: List[PageElement], gold_elements: List[PageElement], threshold: float = 0.5) -> Dict[str, float]:
        """
        计算元素级别 (BBox) 的 Precision, Recall 和 F1。
        匹配逻辑：必须在同一页面 (corpus_id 匹配) 且 IoU 超过阈值。
        """
        if not gold_elements:
            return {}
        if not pred_elements:
            return {"element_precision": 1.0, "element_recall": 0.0, "element_f1": 0.0}

        def normalize_cid(cid):
            return os.path.basename(cid) if cid else ""

        # 计算 Precision: 预测出的元素有多少是命中的
        hit_preds = 0
        for p in pred_elements:
            p_cid = normalize_cid(p.corpus_id)
            for g in gold_elements:
                g_cid = normalize_cid(g.corpus_id)
                # 跨页面匹配校验：只有在同一个页面上的框才计算 IoU
                if p_cid == g_cid:
                    # 使用 base_loader 中定义的 calc_iou_standard
                    if self.calc_iou_standard(p.bbox, g.bbox) > threshold:
                        hit_preds += 1
                        break
        precision = hit_preds / len(pred_elements) if pred_elements else 1.0

        # 计算 Recall: 真值元素有多少被找回了
        hit_golds = 0
        for g in gold_elements:
            g_cid = normalize_cid(g.corpus_id)
            for p in pred_elements:
                p_cid = normalize_cid(p.corpus_id)
                if p_cid == g_cid:
                    if calc_iou_standard(p.bbox, g.bbox) > threshold:
                        hit_golds += 1
                        break
        recall = hit_golds / len(gold_elements) if gold_elements else 1.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"element_precision": precision, "element_recall": recall, "element_f1": f1}

    # --------------------------------------------------------------------------------
    # 更新后的 evaluate 方法
    # --------------------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估：包含模型回答、页面检索指标和元素提取指标。
        """        
        total_metrics = {
            "model_eval": 0,
            "page_recall": 0, "page_precision": 0,
            "element_recall": 0, "element_precision": 0, "element_f1": 0
        }
        counts = {
            "total": 0, "page": 0, "element": 0
        }

        print(f"Starting Evaluation on {len(self.samples)} samples...")
        
        for sample in tqdm(self.samples, desc="Evaluating"):
            if sample.extra_info is None:
                sample.extra_info = {}
            
            # 1. 模型答案准确性 (LLM Judge)
            corr_metrics = self._evaluate_answer_correctness(sample)
            total_metrics['model_eval'] += corr_metrics['model_eval']
            counts['total'] += 1

            # 获取预测的元素列表
            retrieved_elements = sample.extra_info.get('retrieved_elements', [])
            # 统一转为 PageElement 对象以使用其属性
            elements_obj = []
            for el in retrieved_elements:
                if isinstance(el, dict):
                     valid_keys = PageElement.__annotations__.keys()
                     pe = PageElement(**{k:v for k,v in el.items() if k in valid_keys})
                     elements_obj.append(pe)
                elif isinstance(el, PageElement):
                     elements_obj.append(el)

            # 2. 页面指标计算 (Page Precision/Recall)
            # gold_pages 通常存在于 sample.gold_pages 或 extra_info['from_pages']
            target_gold_pages = sample.gold_pages if sample.gold_pages else sample.extra_info.get('from_pages', [])
            if target_gold_pages:
                page_res = self._compute_page_metrics(elements_obj, target_gold_pages)
                total_metrics['page_recall'] += page_res['recall']
                total_metrics['page_precision'] += page_res['precision']
                counts['page'] += 1

            # 3. 元素指标计算 (BBox Precision/Recall/F1)
            # 只有当样本包含标注的 gold_elements 时才进行统计
            if sample.gold_elements:
                elem_res = self._compute_element_metrics(elements_obj, sample.gold_elements)
                if elem_res:
                    total_metrics['element_recall'] += elem_res['element_recall']
                    total_metrics['element_precision'] += elem_res['element_precision']
                    total_metrics['element_f1'] += elem_res['element_f1']
                    counts['element'] += 1

        # 计算平均分
        avg_results = {}
        if counts['total'] > 0:
            avg_results['avg_model_eval'] = total_metrics['model_eval'] / counts['total']
        if counts['page'] > 0:
            avg_results['avg_page_recall'] = total_metrics['page_recall'] / counts['page']
            avg_results['avg_page_precision'] = total_metrics['page_precision'] / counts['page']
        if counts['element'] > 0:
            avg_results['avg_element_recall'] = total_metrics['element_recall'] / counts['element']
            avg_results['avg_element_precision'] = total_metrics['element_precision'] / counts['element']
            avg_results['avg_element_f1'] = total_metrics['element_f1'] / counts['element']

        print(f"Evaluation Results: {avg_results}")
        return avg_results

if __name__ == "__main__":
    # 配置路径
    embedding_model_path = "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/Qwen/Qwen3-VL-Embedding-8B"
    reranker_model_path = "http://localhost:8003"
    root_dir = "/mnt/shared-storage-user/mineru2-shared/jiayu/data/FinRAGBench-V"
    
    # 模拟工具和 Extractor 的初始化参数
    from src.agents.utils import ImageZoomOCRTool
    tool_work_dir = "./workspace" 
    
    print("Initializing Models...")
    embedder = Qwen3VLEmbedder(model_name_or_path=embedding_model_path, torch_dtype=torch.float16)
    reranker = Qwen3VLReranker(model_name_or_path=reranker_model_path, torch_dtype=torch.float16)
    
    tool = ImageZoomOCRTool(work_dir=tool_work_dir)
    extractor = ElementExtractor(
        base_url="http://localhost:8001/v1",
        api_key="sk-123456",
        model_name="MinerU-Agent-CK800",
        tool=tool
    )

    loader = FinRAGLoader(
        data_root=root_dir, 
        lang="ch", 
        embedding_model=embedder, 
        rerank_model=reranker,
        extractor=extractor
    )
    
    loader.llm_caller = create_llm_caller()
    
    loader.load_data()
    
    if len(loader.samples) > 0:
        test_sample = loader.samples[0]
        print(f"\nTesting Query: {test_sample.query}")
        
        # 运行 pipeline 并保存结果到 extra_info
        results = loader.pipeline(test_sample.query, image_paths=[test_sample.data_source], top_k=10) 
        test_sample.extra_info['final_answer'] = "Generated Answer Here..." # 模拟生成答案
        test_sample.extra_info['retrieved_elements'] = results
        
        loader.evaluate()