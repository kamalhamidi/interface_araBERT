import streamlit as st
from transformers import pipeline
import torch

# Page configuration
st.set_page_config(
    page_title="نظام الأسئلة والأجوبة العربي",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, readable design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
    
    * {
        font-family: 'Tajawal', sans-serif;
    }
    
    .main {
        direction: rtl;
        text-align: right;
        background: #f8f9fa;
    }
    
    .stApp {
        background: #f8f9fa;
    }
    
    .stTextArea textarea, .stTextInput input {
        direction: rtl;
        text-align: right;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        font-size: 16px;
        padding: 15px;
        background: white;
        color: #212529;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #0d6efd;
        box-shadow: 0 0 0 3px rgba(13, 110, 253, 0.1);
    }
    
    .stButton button {
        background: #0d6efd;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 18px;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    
    .stButton button:hover {
        background: #0b5ed7;
    }
    
    .answer-card {
        background: white;
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #0d6efd;
        color: #212529;
        font-size: 20px;
        line-height: 1.8;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .title-main {
        text-align: center;
        color: #212529;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 10px;
        margin-top: 20px;
    }
    
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 18px;
        margin-bottom: 40px;
    }
    
    .section-title {
        color: #212529;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
        margin-top: 25px;
    }
    
    .confidence-badge {
        background: #d1e7dd;
        color: #0f5132;
        padding: 8px 20px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        font-size: 16px;
        margin-top: 15px;
        border: 1px solid #badbcc;
    }
    
    .example-card {
        background: white;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.2s;
        cursor: pointer;
    }
    
    .example-card:hover {
        border-color: #0d6efd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    hr {
        border: none;
        height: 1px;
        background: #dee2e6;
        margin: 30px 0;
    }
    
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 30px;
        font-size: 14px;
    }
    
    .history-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .history-card:hover {
        border-color: #0d6efd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .history-question {
        color: #212529;
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 8px;
    }
    
    .history-answer {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .history-time {
        color: #adb5bd;
        font-size: 12px;
    }
    
    .clear-history-btn {
        background: #dc3545;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 14px;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .clear-history-btn:hover {
        background: #bb2d3b;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='title-main'>📚 نظام الأسئلة والأجوبة العربي</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>اطرح سؤالك واحصل على الإجابة من النص</p>", unsafe_allow_html=True)

# Cache the model
@st.cache_resource
def load_qa_model():
    """Load the Arabic QA model"""
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model="ouabdelkrimmina/Arabic-QA-AraBERT",
            device=0 if torch.cuda.is_available() else -1
        )
        return qa_pipeline
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None

# Example data
EXAMPLES = [
    {
        "title": "💼 السيرة الذاتية",
        "context": "محمد بن سلمان هو ولي العهد السعودي ونائب رئيس مجلس الوزراء ووزير الدفاع. ولد في 31 أغسطس 1985 في جدة. يعتبر المهندس الرئيسي لرؤية السعودية 2030، وهي خطة طموحة لتنويع الاقتصاد السعودي وتقليل الاعتماد على النفط.",
        "question": "متى ولد محمد بن سلمان؟"
    },
    {
        "title": "🔬 العلوم",
        "context": "الماء هو مركب كيميائي يتكون من ذرتين من الهيدروجين وذرة واحدة من الأكسجين. يغطي الماء حوالي 71% من سطح الأرض. درجة غليان الماء هي 100 درجة مئوية عند مستوى سطح البحر ويتجمد عند درجة صفر مئوية.",
        "question": "كم نسبة تغطية الماء لسطح الأرض؟"
    },
    {
        "title": "📖 التاريخ",
        "context": "تأسست المملكة العربية السعودية عام 1932 على يد الملك عبد العزيز آل سعود بعد توحيد مناطق شبه الجزيرة العربية. تعتبر المملكة من أكبر الدول المنتجة للنفط في العالم وتحتوي على أقدس المواقع الإسلامية.",
        "question": "من أسس المملكة العربية السعودية؟"
    }
]

# Initialize session state
if 'context' not in st.session_state:
    st.session_state.context = ""
if 'question' not in st.session_state:
    st.session_state.question = ""
if 'history' not in st.session_state:
    st.session_state.history = []

# Main container
with st.container():
    # Examples section
    st.markdown("<p class='section-title'>جرب الأمثلة التالية:</p>", unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, example in enumerate(EXAMPLES):
        with cols[idx]:
            if st.button(example["title"], key=f"example_{idx}", use_container_width=True):
                st.session_state.context = example["context"]
                st.session_state.question = example["question"]
                st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Context input
    st.markdown("<p class='section-title'>النص:</p>", unsafe_allow_html=True)
    context = st.text_area(
        "context",
        value=st.session_state.context,
        height=150,
        placeholder="أدخل النص الذي تريد طرح سؤال عنه...",
        label_visibility="collapsed"
    )
    
    # Question input
    st.markdown("<p class='section-title'>السؤال:</p>", unsafe_allow_html=True)
    question = st.text_input(
        "question",
        value=st.session_state.question,
        placeholder="اكتب سؤالك هنا...",
        label_visibility="collapsed"
    )
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 احصل على الإجابة", use_container_width=True):
        if not context.strip() or not question.strip():
            st.warning("⚠️ الرجاء إدخال النص والسؤال")
        else:
            with st.spinner("جاري البحث عن الإجابة..."):
                qa_model = load_qa_model()
                
                if qa_model:
                    try:
                        result = qa_model(question=question, context=context)
                        
                        # Save to history
                        from datetime import datetime
                        history_item = {
                            'question': question,
                            'answer': result['answer'],
                            'context': context,
                            'confidence': result['score'] * 100,
                            'time': datetime.now().strftime("%Y-%m-%d %H:%M")
                        }
                        st.session_state.history.insert(0, history_item)
                        
                        # Keep only last 10 items
                        if len(st.session_state.history) > 10:
                            st.session_state.history = st.session_state.history[:10]
                        
                        # Display answer
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown("<p class='section-title'>الإجابة:</p>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div class='answer-card'><strong>{result['answer']}</strong></div>",
                            unsafe_allow_html=True
                        )
                        
                        # Confidence score
                        confidence = result['score'] * 100
                        st.markdown(
                            f"<div style='text-align: center;'><span class='confidence-badge'>درجة الثقة: {confidence:.1f}%</span></div>",
                            unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ حدث خطأ: {str(e)}")

# History section
if st.session_state.history:
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<p class='section-title'>📋 السجل (آخر 10 أسئلة):</p>", unsafe_allow_html=True)
    with col2:
        if st.button("مسح السجل", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    
    for idx, item in enumerate(st.session_state.history):
        with st.container():
            if st.button(
                f" {item['question'][:60]}{'...' if len(item['question']) > 60 else ''}",
                key=f"history_{idx}",
                use_container_width=True
            ):
                st.session_state.context = item['context']
                st.session_state.question = item['question']
                st.rerun()
            
            st.markdown(f"""
                <div style='background: #f8f9fa; padding: 10px; border-radius: 6px; margin-bottom: 10px;'>
                    <div class='history-answer'>✅ {item['answer']}</div>
                    <div class='history-time'>🕐 {item['time']} | درجة الثقة: {item['confidence']:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div class='footer'>
    <p>مدعوم بنموذج AraBERT المدرب على بيانات عربية</p>
</div>
""", unsafe_allow_html=True)