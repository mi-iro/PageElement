import streamlit as st
import json
import os
from PIL import Image, ImageDraw
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"读取JSON文件失败: {e}")
        return None

def draw_bbox_on_image(image_path, bbox):
    """在图片上绘制BBox"""
    if not os.path.exists(image_path):
        return None, f"图片文件未找到: {image_path}"
    
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        
        if bbox and len(bbox) == 4:
            abs_xmin = (bbox[0] / 1000.0) * width
            abs_ymin = (bbox[1] / 1000.0) * height
            abs_xmax = (bbox[2] / 1000.0) * width
            abs_ymax = (bbox[3] / 1000.0) * height
            
            draw.rectangle(
                [abs_xmin, abs_ymin, abs_xmax, abs_ymax], 
                outline="red", 
                width=max(3, int(min(width, height) * 0.005))
            )
        return image, None
    except Exception as e:
        return None, f"处理图片时出错: {e}"

def main():
    st.set_page_config(layout="wide", page_title="RAG 结果可视化")
    st.title("📊 RAG 实验结果可视化")
    
    # --- 侧边栏：配置 ---
    st.sidebar.header("配置")
    base_dir = st.sidebar.text_input("JSON文件目录:", value=os.getcwd())
    
    json_files = []
    if os.path.isdir(base_dir):
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        json_files.sort()
    else:
        st.sidebar.error("无效的目录路径")
        st.stop()
        
    if not json_files:
        st.sidebar.warning("该目录下没有找到JSON文件")
        st.stop()

    # --- 核心修复：状态管理 ---
    
    # 1. 初始化索引
    if 'file_index' not in st.session_state:
        st.session_state.file_index = 0

    # 2. 定义回调函数 (修复点：同时更新 index 和 selector 的值)
    def prev_file():
        if st.session_state.file_index > 0:
            st.session_state.file_index -= 1
            # 强制同步更新下拉框的值
            st.session_state.file_selector = json_files[st.session_state.file_index]

    def next_file():
        if st.session_state.file_index < len(json_files) - 1:
            st.session_state.file_index += 1
            # 强制同步更新下拉框的值
            st.session_state.file_selector = json_files[st.session_state.file_index]

    def on_selector_change():
        # 当用户手动选择下拉框时，反向更新 index
        selected = st.session_state.file_selector
        if selected in json_files:
            st.session_state.file_index = json_files.index(selected)

    # 3. 导航按钮区域
    st.sidebar.markdown("---")
    st.sidebar.subheader("样本切换")
    col_prev, col_info, col_next = st.sidebar.columns([1, 2, 1])
    
    with col_prev:
        st.button("⬅️", on_click=prev_file, disabled=(st.session_state.file_index == 0))
    
    with col_info:
        st.markdown(f"<div style='text-align: center; line-height: 2.2;'>{st.session_state.file_index + 1} / {len(json_files)}</div>", unsafe_allow_html=True)
    
    with col_next:
        st.button("➡️", on_click=next_file, disabled=(st.session_state.file_index == len(json_files) - 1))

    # 4. 文件选择框 (绑定 key 和 on_change)
    # 注意：这里去掉了 index 参数，完全依赖 session_state['file_selector'] 来控制选中项
    
    # 首次加载或 key 不存在时，初始化 selectbox 的值
    if 'file_selector' not in st.session_state:
        st.session_state.file_selector = json_files[st.session_state.file_index]

    selected_file = st.sidebar.selectbox(
        "跳转到文件:", 
        json_files,
        format_func=lambda x: os.path.relpath(x, base_dir),
        key='file_selector',      # 绑定 key
        on_change=on_selector_change # 绑定手动变更的回调
    )
    
    st.sidebar.markdown("---")

    # --- 内容展示 ---
    if selected_file:
        data = load_json(selected_file)
        if not data:
            st.stop()
            
        # 显示基础信息
        with st.expander("📝 基础信息 (Query & Answer)", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**QID:** `{data.get('qid', 'N/A')}`")
                st.info(f"**Query:**\n\n{data.get('query', 'N/A')}")
            with col2:
                st.success(f"**Gold Answer:**\n\n{data.get('gold_answer', 'N/A')}")
                st.warning(f"**Final Answer:**\n\n{data.get('final_answer', 'N/A')}")

        # 显示对话
        st.header("💬 对话历史")
        for idx, msg in enumerate(data.get('messages', [])):
            with st.chat_message(msg.get('role', 'user')):
                st.write(f"**[{idx}] {msg.get('role')}**")
                content = msg.get('content')
                if isinstance(content, str):
                    st.markdown(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get('type') == 'text': st.markdown(item.get('text'))
                        elif item.get('type') == 'image_url': 
                            st.image(item['image_url']['url'], width=300)

        st.divider()

        # 显示检索结果
        st.header("🔍 检索结果")
        for i, elem in enumerate(data.get('retrieved_elements', [])):
            st.subheader(f"Evidence #{i+1}")
            col_text, col_img = st.columns([1, 1])
            
            with col_text:
                st.text_area("Content", elem.get('content', ''), height=200, key=f"txt_{i}_{selected_file}")
                with st.expander("Metadata"):
                    st.json({k:v for k,v in elem.items() if k not in ['content']})
            
            with col_img:
                if elem.get('corpus_path'):
                    img, err = draw_bbox_on_image(elem['corpus_path'], elem.get('bbox'))
                    if img: st.image(img, caption=f"Source: {os.path.basename(elem['corpus_path'])}")
                    else: st.error(err)
            st.divider()

if __name__ == "__main__":
    main()