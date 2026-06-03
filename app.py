import streamlit as st
import joblib

# Page Config
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.result-box {
    padding: 15px;
    border-radius: 10px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Load Model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# Heading
st.markdown('<div class="title">📩 Spam Message Detector</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Check whether a message is Spam or Normal</div>',
    unsafe_allow_html=True
)

# Input Box
message = st.text_area(
    "Enter Message",
    height=150,
    placeholder="Type your message here..."
)

# Button
if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([message])
        prediction = model.predict(msg_vec)[0]

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE DETECTED")
        else:
            st.success("✅ NORMAL MESSAGE")

# Footer
st.markdown("---")
st.caption("Built by Chandra Prakash Choubisa!🙏")