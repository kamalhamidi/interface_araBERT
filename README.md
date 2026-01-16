# 📚 نظام الأسئلة والأجوبة العربي | Arabic Question Answering System

نظام ذكي للإجابة على الأسئلة باللغة العربية باستخدام نموذج AraBERT المدرب على مجموعة بيانات Arabic SQuAD.

A smart system for answering questions in Arabic using AraBERT model fine-tuned on Arabic SQuAD dataset.

![Arabic QA System](https://img.shields.io/badge/Language-Arabic-green)
![Model](https://img.shields.io/badge/Model-AraBERT-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red)

## ✨ المميزات | Features

- 🤖 **نموذج متقدم**: يستخدم AraBERT المدرب خصيصاً على نصوص عربية
- 📝 **واجهة بسيطة**: تصميم نظيف وسهل الاستخدام
- 📋 **سجل الأسئلة**: حفظ آخر 10 أسئلة وإجاباتها
- 🎯 **درجة الثقة**: عرض مستوى ثقة النموذج في الإجابة
- 🔄 **أمثلة جاهزة**: ثلاثة أمثلة للتجربة السريعة
- 🌐 **دعم كامل للعربية**: واجهة من اليمين لليسار (RTL)

---

## 🚀 البدء السريع | Quick Start

### المتطلبات | Prerequisites

```bash
Python 3.8+
```

### التثبيت | Installation

1. **استنساخ المشروع | Clone the repository**

```bash
git clone https://github.com/kamalhamidi/interface_araBERT/
cd arabic-qa-system
```

2. **تثبيت المكتبات المطلوبة | Install dependencies**

```bash
pip install -r requirements.txt
```

3. **تشغيل التطبيق | Run the application**

```bash
streamlit run app.py
```

4. **فتح المتصفح | Open your browser**

افتح المتصفح على: `http://localhost:8501`

---

## 📦 الملفات المطلوبة | Required Files

### requirements.txt

```
streamlit==1.28.0
transformers==4.35.0
torch==2.1.0
```

### structure المشروع | Project Structure

```
arabic-qa-system/
│
├── app.py                 # التطبيق الرئيسي | Main application
├── requirements.txt       # المكتبات المطلوبة | Dependencies
├── README.md             # هذا الملف | This file
└── .gitignore           # ملفات Git | Git ignore file
```

---

## 🎯 كيفية الاستخدام | How to Use

### 1. إدخال النص | Enter Text
أدخل النص العربي الذي تريد طرح سؤال عنه في حقل "النص"

Enter the Arabic text you want to ask questions about in the "Text" field

### 2. كتابة السؤال | Write Question
اكتب سؤالك المتعلق بالنص في حقل "السؤال"

Write your question related to the text in the "Question" field

### 3. الحصول على الإجابة | Get Answer
اضغط على زر "احصل على الإجابة" وانتظر النتيجة

Click the "Get Answer" button and wait for the result

### 4. استخدام الأمثلة | Use Examples
جرب الأمثلة الجاهزة بالضغط على أحد الأزرار في الأعلى

Try the ready examples by clicking one of the buttons at the top

### 5. مراجعة السجل | Review History
راجع آخر 10 أسئلة في قسم السجل أسفل الصفحة

Review the last 10 questions in the history section at the bottom

---

## 🤖 النموذج المستخدم | Model Used

**Model**: [ouabdelkrimmina/Arabic-QA-AraBERT](https://huggingface.co/ouabdelkrimmina/Arabic-QA-AraBERT)

- **Base Model**: AraBERTv2
- **Task**: Question Answering
- **Training Data**: Arabic SQuAD
- **Language**: Arabic

---

## 📊 أمثلة | Examples

### مثال 1: السيرة الذاتية

**النص:**
```
محمد بن سلمان هو ولي العهد السعودي ونائب رئيس مجلس الوزراء ووزير الدفاع. 
ولد في 31 أغسطس 1985 في جدة.
```

**السؤال:** متى ولد محمد بن سلمان؟

**الإجابة:** 31 أغسطس 1985

### مثال 2: العلوم

**النص:**
```
الماء هو مركب كيميائي يتكون من ذرتين من الهيدروجين وذرة واحدة من الأكسجين. 
يغطي الماء حوالي 71% من سطح الأرض.
```

**السؤال:** كم نسبة تغطية الماء لسطح الأرض؟

**الإجابة:** 71%

---

## 🌐 النشر على Hugging Face Spaces | Deploy to Hugging Face Spaces

### 1. إنشاء Space جديد | Create New Space

1. اذهب إلى [Hugging Face Spaces](https://huggingface.co/spaces)
2. اضغط على "Create new Space"
3. اختر اسم المشروع واختر SDK: **Streamlit**

### 2. رفع الملفات | Upload Files

ارفع الملفات التالية:
- `app.py`
- `requirements.txt`
- `README.md`

### 3. إضافة ملف README.md لـ Space

أنشئ ملف `README.md` في Space بالمحتوى التالي:

```markdown
---
title: Arabic QA System
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

Check configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
```

---

## 🛠️ التخصيص | Customization

### تغيير النموذج | Change Model

في ملف `app.py`، استبدل اسم النموذج:

```python
qa_pipeline = pipeline(
    "question-answering",
    model="YOUR_MODEL_NAME_HERE",  # غير هنا
    device=0 if torch.cuda.is_available() else -1
)
```

### تغيير الألوان | Change Colors

عدّل قسم CSS في `app.py` لتغيير الألوان:

```python
.stButton button {
    background: #0d6efd;  # لون الزر
}

.answer-card {
    border: 2px solid #0d6efd;  # لون إطار الإجابة
}
```

### إضافة أمثلة جديدة | Add New Examples

أضف أمثلة جديدة في قائمة `EXAMPLES`:

```python
EXAMPLES = [
    {
        "title": "🎨 عنوان المثال",
        "context": "النص هنا...",
        "question": "السؤال هنا؟"
    },
    # أضف المزيد...
]
```

---

## 🐛 حل المشاكل | Troubleshooting

### المشكلة: النموذج لا يتحمل

**الحل:**
- تأكد من اتصالك بالإنترنت
- تحقق من اسم النموذج الصحيح
- حاول تشغيل الأمر: `pip install --upgrade transformers`

### المشكلة: الإجابات غير دقيقة

**الحل:**
- تأكد من وضوح السؤال
- استخدم نصوص ذات صلة بالسؤال
- جرب إعادة صياغة السؤال

### المشكلة: بطء في الاستجابة

**الحل:**
- استخدم نصوص أقصر
- إذا كان لديك GPU، تأكد من تفعيله
- قلل من طول النص المدخل

---

## 📝 الترخيص | License

هذا المشروع مرخص تحت رخصة MIT - انظر ملف [LICENSE](LICENSE) للتفاصيل

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

## 🤝 المساهمة | Contributing

المساهمات مرحب بها! يرجى:

1. عمل Fork للمشروع
2. إنشاء فرع جديد (`git checkout -b feature/AmazingFeature`)
3. إجراء التعديلات (`git commit -m 'Add some AmazingFeature'`)
4. رفع التعديلات (`git push origin feature/AmazingFeature`)
5. فتح Pull Request

---

## 📧 التواصل | Contact

إذا كان لديك أي أسئلة أو اقتراحات، لا تتردد في التواصل!

If you have any questions or suggestions, feel free to reach out!

---

## 🙏 شكر وتقدير | Acknowledgments

- [Hugging Face](https://huggingface.co/) لتوفير منصة النماذج
- [AraBERT](https://github.com/aub-mind/arabert) لنموذج AraBERT
- [Streamlit](https://streamlit.io/) لإطار العمل الرائع
- مجموعة بيانات Arabic SQuAD

---

## 📊 الإحصائيات | Stats

- **اللغة**: العربية | Arabic
- **النموذج**: AraBERT
- **الدقة**: تعتمد على النموذج المستخدم
- **الاستجابة**: 1-3 ثواني

---

<div align="center">
  <p>صنع بـ ❤️ للغة العربية</p>
  <p>Made with ❤️ for Arabic Language</p>
</div>
