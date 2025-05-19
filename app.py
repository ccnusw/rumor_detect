import streamlit as st
import os
from openai import OpenAI

# 设置页面配置
st.set_page_config(layout="wide", page_title="网络新闻标题失范自动检测系统")

# --- API Configuration via Sidebar --- #
client = None
final_api_key = None
final_base_url = None
final_model_name = None

with st.sidebar.expander("🔑 API密钥配置", expanded=True):
    st.write("请在此处配置您的API密钥和模型。")
    api_provider = st.radio(
        "选择API提供商:",
        ('OpenRouter', 'Qwen官方', '使用环境变量 OPENROUTER_API_KEY'),
        index=0,
        help="选择您的API服务提供商。如果您选择'使用环境变量'，请确保已设置 OPENROUTER_API_KEY。"
    )

    if api_provider == 'OpenRouter' or api_provider == 'Qwen官方':
        manual_api_key = st.text_input(
            "输入您的API密钥:",
            type="password",
            help="请输入您的API密钥。"
        )
        if manual_api_key:
            final_api_key = manual_api_key
            if api_provider == 'OpenRouter':
                final_base_url = "https://openrouter.ai/api/v1"
                final_model_name = st.text_input("OpenRouter 模型名称:", value="qwen/qwen3-235b-a22b:free", help="例如: qwen/qwen3-235b-a22b:free")
                st.caption(f"使用: 手动输入的 OpenRouter 密钥 (模型: {final_model_name})")
            elif api_provider == 'Qwen官方':
                final_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" # Qwen官方兼容OpenAI的endpoint
                final_model_name = st.text_input("Qwen 模型名称:", value="qwen-turbo", help="例如: qwen-turbo, qwen-max")
                st.caption(f"使用: 手动输入的 Qwen 密钥 (模型: {final_model_name})")
        else:
            st.warning("请输入API密钥。")

    elif api_provider == '使用环境变量 OPENROUTER_API_KEY':
        env_api_key = os.environ.get("OPENROUTER_API_KEY")
        if env_api_key:
            final_api_key = env_api_key
            final_base_url = "https://openrouter.ai/api/v1"
            final_model_name = "qwen/qwen3-235b-a22b:free" # 默认使用用户指定的OpenRouter模型
            st.caption(f"使用: 环境变量 OPENROUTER_API_KEY (模型: {final_model_name})")
        else:
            st.error("未找到 OPENROUTER_API_KEY 环境变量。请在上方选择其他方式或设置该环境变量。")

# 初始化OpenAI客户端
# 尝试初始化客户端，但不在此处停止应用
if final_api_key and final_base_url and final_model_name:
    try:
        client = OpenAI(
            base_url=final_base_url,
            api_key=final_api_key,
        )
        # st.sidebar.success("API客户端已准备就绪。") # 暂时注释掉，避免在未点击分析时显示
    except Exception as e:
        # st.error(f"初始化OpenAI客户端失败: {e}") # 暂时注释掉，避免在未点击分析时显示
        client = None # 确保客户端为None，以便后续检查
# 不再在此处强制停止应用，将检查移到按钮点击事件中


# 全局变量，用于存储上次的输入和结果，避免重复请求
if 'last_title' not in st.session_state:
    st.session_state.last_title = ""
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

def analyze_title(title: str, current_client: OpenAI, model_to_use: str):
    """
    使用大语言模型分析新闻标题。

    参数:
        title (str): 用户输入的新闻标题。
        current_client (OpenAI): 配置好的OpenAI API客户端。
        model_to_use (str): 要使用的模型名称。

    返回:
        dict: 包含分析结果的字典，例如：
              {
                  "probability": 0.85, # 疑似标题党概率
                  "suggestions": "建议修改标题以更客观地反映内容...",
                  "modified_title": "修改后的标题..." # 如果概率 > 0.6
              }
              如果API调用失败，则返回None。
    """
    if not current_client:
        st.error("API客户端未初始化，无法分析标题。")
        return None
        
    if not title.strip():
        return {
            "probability": 0.0,
            "suggestions": "请输入新闻标题进行分析。",
            "modified_title": ""
        }

    # 避免对相同的标题重复请求 (如果模型或客户端也作为缓存键的一部分会更好，但这里简化)
    if title == st.session_state.last_title and st.session_state.last_result is not None:
        # 简单起见，如果标题相同就返回上次结果。实际应用可能需要更复杂的缓存策略
        # 特别是如果API Key或模型改变了，应该重新请求
        pass # 允许重新请求，因为API配置可能已更改

    # 构建提示
    prompt_text = f"""
    作为一名经验丰富的标题研究专家，请分析以下新闻标题：'{title}'

    请评估其作为“标题党”或谣言标题的可能性，并给出以下信息：
    1. 疑似“标题党”的概率（0.0到1.0之间的一个浮点数）。
    2. 具体的修改建议，使其更符合新闻规范。
    3. 如果疑似“标题党”的概率达到或超过50%（即概率 >= 0.5），请给出一个修改后的、更客观中立的标题建议。

    请严格按照以下JSON格式返回结果，不要添加任何额外的解释或说明文字：
    {{
        "probability": <float_value>,
        "suggestions": "<string_value>",
        "modified_title": "<string_value_or_empty_if_not_applicable>"
    }}

    谣言标题常见的套路包括：
    - 伪装权威来源 (例如：“中央发话了”, “央视都播了”)
    - 捏造夸大事实 (例如：“震惊！”, “惊爆！”, “出大事了！！！”)
    - “煲鸡汤”煽情型 (例如：“为了你的家人，请一定要看”)
    - 求阅读转发 (例如：“速看、马上删”, “警惕、紧急通知”)

    请基于以上信息进行判断。
    """

    import urllib.parse # 导入urllib.parse模块
    try:
        encoded_x_title = urllib.parse.quote("网络新闻标题失范自动检测系统")
        completion = current_client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://news-title-checker.streamlit.app", # 替换为您的实际部署URL
                "X-Title": encoded_x_title, # 使用URL编码后的X-Title
            },
            model=model_to_use, # 使用传入的模型名称
            messages=[
                {
                    "role": "system",
                    "content": "你是一名经验丰富的标题研究专家，精通标题党的知识和技巧。你的任务是分析新闻标题，并以JSON格式返回分析结果。"
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        response_content = completion.choices[0].message.content
        import json
        import re
        try:
            # 尝试去除Markdown代码块标记
            match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
            if match:
                cleaned_response_content = match.group(1).strip()
            else:
                # 如果没有找到Markdown代码块，尝试去除可能存在的单个```
                cleaned_response_content = response_content.strip()
                if cleaned_response_content.startswith("```json"):
                    cleaned_response_content = cleaned_response_content[len("```json"):].strip()
                if cleaned_response_content.startswith("```"):
                    cleaned_response_content = cleaned_response_content[len("```"):].strip()
                if cleaned_response_content.endswith("```"):
                    cleaned_response_content = cleaned_response_content[:-len("```")].strip()
            
            result = json.loads(cleaned_response_content)
            if not all(k in result for k in ["probability", "suggestions", "modified_title"]):
                st.error(f"大语言模型返回的JSON格式不符合预期。原始返回内容：'{response_content}'，清理后内容：'{cleaned_response_content}'")
                return None
            st.session_state.last_title = title
            st.session_state.last_result = result
            return result
        except json.JSONDecodeError:
            st.error(f"无法解析大语言模型返回的JSON。原始返回内容：'{response_content}'，清理后尝试解析的内容：'{cleaned_response_content}'")
            # 当解析失败时，也尝试返回一个结构，以便UI可以处理
            return {
                "probability": 0.0, # 或者一个特殊值来指示错误
                "suggestions": f"模型返回格式错误，无法解析JSON。原始输出: {response_content}",
                "modified_title": ""
            }

    except Exception as e:
        st.error(f"调用大语言模型API时出错: {e}")
        return None

# --- Streamlit UI --- #

# 标题
st.markdown("<h1 style='text-align: center; font-size: 28px;'>🌐 网络新闻标题失范自动检测系统</h1>", unsafe_allow_html=True)

# 输入区域
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown("**请输入新闻标题进行检测：**") # 将subheader改为markdown并加粗，使其显示在输入框上方
    title_input = st.text_input("", placeholder="例如：震惊！科学家发现...", label_visibility="collapsed")
    analyze_button = st.button("🔍 分析标题")


# 结果显示区域
if analyze_button and title_input:
    # 首先检查API配置是否完整
    if not final_api_key or not final_base_url or not final_model_name:
        st.error("API密钥、基础URL或模型名称未配置。请在侧边栏中正确配置API信息后再试。")
    else:
        # 尝试初始化或重新确认客户端
        if client is None: # 如果之前初始化失败或未初始化
            try:
                client = OpenAI(
                    base_url=final_base_url,
                    api_key=final_api_key,
                )
                st.sidebar.success("API客户端已成功初始化并准备就绪。")
            except Exception as e:
                st.error(f"初始化OpenAI客户端失败: {e}。请检查API配置。")
                client = None # 确保客户端为None

        if client and final_model_name: # 再次确保客户端和模型名称已配置
            with st.spinner("正在分析中，请稍候..."):
                analysis_result = analyze_title(title_input, client, final_model_name)
            
            if analysis_result:
                st.subheader("检测结果：")
                
                probability = analysis_result.get("probability", 0.0)
                suggestions_from_model = analysis_result.get("suggestions", "未能获取建议。")
                modified_title_from_model = analysis_result.get("modified_title", "")

                # 根据概率调整建议和修改后的标题
                if isinstance(probability, (float, int)):
                    if probability < 0.5:
                        suggestions_to_display = "不像标题党，不用修改。"
                        modified_title_to_display = ""
                    else:
                        suggestions_to_display = suggestions_from_model
                        modified_title_to_display = modified_title_from_model

                    # 显示概率
                    if probability > 0.7:
                        st.error(f"**疑似“标题党”概率：{probability*100:.2f}%**")
                    elif probability >= 0.5: # 从0.4改为0.5，与新逻辑一致
                        st.warning(f"**疑似“标题党”概率：{probability*100:.2f}%**")
                    else: # probability < 0.5
                        st.success(f"**疑似“标题党”概率：{probability*100:.2f}%**")
                else:
                    st.warning(f"概率值格式不正确: {probability}")
                    suggestions_to_display = suggestions_from_model # 如果概率格式不对，还是显示原始建议
                    modified_title_to_display = modified_title_from_model

                st.markdown("**修改建议：**")
                st.info(suggestions_to_display)
                
                # 仅当概率 >= 0.5 且有修改后的标题时才显示
                if modified_title_to_display and isinstance(probability, (float, int)) and probability >= 0.5:
                    st.markdown("**修改后的标题参考：**")
                    st.success(modified_title_to_display)
            elif title_input: 
                st.error("分析失败或未能获取结果，请检查API密钥或网络连接，并稍后重试。")
        elif not client:
             st.error("API客户端未能成功初始化。请检查侧边栏中的API密钥设置。")
        elif not final_model_name:
            st.error("模型名称未确定。请在侧边栏中正确配置模型信息。")


# 页面底部版权信息
st.markdown(
    "<div style='text-align: center; color: grey;'>" 
    "本软件系国家语委项目“网络新闻标题失范的话语表征及对策研究”【YB145-66】阶段性成果。"
    "</div>", 
    unsafe_allow_html=True
)

# 添加一些自定义CSS来美化界面 (可选)
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 提示用户如何运行
st.sidebar.info("请在终端中使用 `streamlit run app.py` 命令来启动此应用。")