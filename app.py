import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Page configuration
st.set_page_config(
    page_title="نظام الأسئلة والأجوبة العربي",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL and styling
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    .stTextArea textarea {
        direction: rtl;
        text-align: right;
    }
    .stTextInput input {
        direction: rtl;
        text-align: right;
    }
    .answer-box {
        padding: 20px;
        background-color: #d4edda;
        border-radius: 10px;
        border: 2px solid #28a745;
        direction: rtl;
        text-align: right;
        font-size: 18px;
        margin: 10px 0;
    }
    .title {
        text-align: center;
        color: #28a745;
        direction: rtl;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🤖 نظام الأسئلة والأجوبة العربي</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='title'>مدعوم بنموذج AraBERT</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for model configuration
with st.sidebar:
    st.header("⚙️ إعدادات النموذج")
    
    model_option = st.radio(
        "اختر طريقة تحميل النموذج:",
        ["نموذج من Hugging Face Hub", "نموذج محلي"]
    )
    
    if model_option == "نموذج من Hugging Face Hub":
        model_name = st.text_input(
            "اسم النموذج:",
            value="aubmindlab/bert-base-arabertv2",
            help="مثال: username/model-name أو aubmindlab/bert-base-arabertv2"
        )
    else:
        model_path = st.text_input(
            "مسار النموذج المحلي:",
            value="./model",
            help="المسار إلى مجلد النموذج على جهازك"
        )
    
    st.markdown("---")
    st.markdown("### 📖 كيفية الاستخدام")
    st.markdown("""
    1. أدخل النص العربي
    2. اكتب سؤالك
    3. اضغط على زر الإجابة
    4. جرب الأمثلة المحملة
    """)

# Cache the model loading
@st.cache_resource
def load_model(model_name_or_path):
    """Load the QA model and tokenizer"""
    try:
        with st.spinner("جاري تحميل النموذج..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
            qa_pipeline = pipeline(
                "question-answering",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            st.success("✅ تم تحميل النموذج بنجاح!")
            return qa_pipeline
    except Exception as e:
        st.error(f"❌ خطأ في تحميل النموذج: {str(e)}")
        return None

# Example data
EXAMPLES = {
    "مثال 1: السيرة الذاتية": {
        "context": "محمد بن سلمان هو ولي العهد السعودي ونائب رئيس مجلس الوزراء ووزير الدفاع. ولد في 31 أغسطس 1985 في جدة. يعتبر المهندس الرئيسي لرؤية السعودية 2030، وهي خطة طموحة لتنويع الاقتصاد السعودي وتقليل الاعتماد على النفط.",
        "question": "متى ولد محمد بن سلمان؟"
    },
    "مثال 2: العلوم": {
        "context": "الماء هو مركب كيميائي يتكون من ذرتين من الهيدروجين وذرة واحدة من الأكسجين. يغطي الماء حوالي 71% من سطح الأرض. درجة غليان الماء هي 100 درجة مئوية عند مستوى سطح البحر.",
        "question": "ما هي نسبة تغطية الماء لسطح الأرض؟"
    },
    "مثال 3: التاريخ": {
        "context": "تأسست الدولة السعودية الأولى عام 1744 على يد محمد بن سعود. وفي عام 1932، تم توحيد المملكة العربية السعودية على يد الملك عبد العزيز آل سعود. تعتبر المملكة من أكبر الدول المنتجة للنفط في العالم.",
        "question": "من قام بتوحيد المملكة العربية السعودية؟"
    }
}

# Example selector
col1, col2 = st.columns([3, 1])
with col2:
    selected_example = st.selectbox(
        "اختر مثالاً:",
        ["لا يوجد"] + list(EXAMPLES.keys())
    )

# Main content area
col_context, col_question = st.columns(2)

with col_context:
    st.subheader("📄 النص (السياق)")
    if selected_example != "لا يوجد":
        context = st.text_area(
            "أدخل النص الذي تريد طرح سؤال عنه:",
            value=EXAMPLES[selected_example]["context"],
            height=200,
            key="context",
            label_visibility="collapsed"
        )
    else:
        context = st.text_area(
            "أدخل النص الذي تريد طرح سؤال عنه:",
            height=200,
            key="context",
            label_visibility="collapsed",
            placeholder="الرجاء إدخال النص هنا..."
        )

with col_question:
    st.subheader("❓ السؤال")
    if selected_example != "لا يوجد":
        question = st.text_input(
            "ما هو سؤالك؟",
            value=EXAMPLES[selected_example]["question"],
            key="question",
            label_visibility="collapsed"
        )
    else:
        question = st.text_input(
            "ما هو سؤالك؟",
            key="question",
            label_visibility="collapsed",
            placeholder="اكتب سؤالك هنا..."
        )

# Get Answer button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    get_answer = st.button("🎯 احصل على الإجابة", use_container_width=True, type="primary")
with col_btn2:
    clear_btn = st.button("🗑️ مسح", use_container_width=True)

if clear_btn:
    st.rerun()

# Process and display answer
if get_answer:
    if not context.strip() or not question.strip():
        st.warning("⚠️ الرجاء إدخال النص والسؤال")
    else:
        # Load model
        if model_option == "نموذج من Hugging Face Hub":
            qa_model = load_model(model_name)
        else:
            qa_model = load_model(model_path)
        
        if qa_model:
            try:
                with st.spinner("🔍 جاري البحث عن الإجابة..."):
                    result = qa_model(question=question, context=context)
                    
                    # Display answer
                    st.markdown("---")
                    st.subheader("✨ الإجابة")
                    st.markdown(
                        f"<div class='answer-box'>{result['answer']}</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Display confidence score
                    col_score1, col_score2, col_score3 = st.columns(3)
                    with col_score2:
                        confidence = result['score'] * 100
                        st.metric("درجة الثقة", f"{confidence:.2f}%")
                    
                    # Display additional info
                    with st.expander("📊 معلومات إضافية"):
                        st.write(f"**موقع الإجابة في النص:** من الحرف {result['start']} إلى {result['end']}")
                        st.write(f"**درجة الثقة الدقيقة:** {result['score']:.4f}")
                        
            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء معالجة السؤال: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; direction: rtl;'>
    <p>تم بناء هذا النظام باستخدام AraBERT و Streamlit</p>
    <p>للحصول على أفضل النتائج، استخدم أسئلة واضحة ومحددة</p>
</div>
""", unsafe_allow_html=True)