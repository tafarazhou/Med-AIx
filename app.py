import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
import base64

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="MEDAIx Investor Demo",
    page_icon="🧠",
    layout="wide"
)

# --- SIDEBAR ---
st.sidebar.title("MEDAIx Investor Dashboard")
page = st.sidebar.radio("Navigation", ["Home", "Upload MRI", "Dashboard", "About MEDAIx"])

# --- COLORS & STYLE ---
primary_color = "#1E90FF"
st.markdown(f"""
<style>
body {{ font-family: 'Segoe UI', sans-serif; }}
.sidebar .sidebar-content {{ background-color: #f8f8f8; }}
</style>
""", unsafe_allow_html=True)

# --- HOME PAGE ---
if page == "Home":
    st.title("🧠 MEDAIx – AI-Powered MRI Diagnostics")
    st.subheader("Transforming MRI analysis with AI insights")
    st.image("https://images.unsplash.com/photo-1581091870620-3b29dca338e4?auto=format&fit=crop&w=1050&q=80", use_column_width=True)
    st.markdown("""
    **Features:**  
    - Multi-condition AI predictions  
    - Heatmap overlays  
    - Interactive dashboard metrics  
    - Investor-ready PDF reports
    """)
    st.markdown("---")

# --- UPLOAD MRI PAGE ---
elif page == "Upload MRI":
    st.header("Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose an MRI image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI", use_column_width=True)
        
        # --- SIMULATED AI PREDICTIONS ---
        st.subheader("AI Predictions")
        conditions = ["Normal", "Tumor", "Stroke", "Degeneration"]
        # Simulate probabilities for each condition
        np.random.seed(42)  # For consistent results
        probabilities = np.random.dirichlet(np.ones(len(conditions)), size=1)[0]
        predictions = dict(zip(conditions, probabilities))
        
        pred_df = pd.DataFrame(
            {"Condition": conditions, "Probability": [f"{p*100:.1f}%" for p in probabilities]}
        )
        st.table(pred_df)
        
        # --- SHOW PREDICTED CLASS ---
        top_pred = conditions[np.argmax(probabilities)]
        st.success(f"**Most likely condition:** {top_pred} ({probabilities[np.argmax(probabilities)]*100:.1f}%)")
        
        # --- SIMULATED HEATMAP ---
        st.subheader("AI Heatmap Overlay")
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        heatmap = np.random.rand(image.size[1], image.size[0])
        ax.imshow(heatmap, cmap='jet', alpha=0.4)
        ax.axis('off')
        st.pyplot(fig)
        
        # --- PDF EXPORT ---
        st.subheader("Download Investor PDF Report")
        def create_pdf(image, predictions):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "MEDAIx Investor MRI Report", ln=True, align="C")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, "AI Condition Probabilities:", ln=True)
            for cond, prob in predictions.items():
                pdf.cell(0, 10, f"{cond}: {prob*100:.1f}%", ln=True)
            # Save image to bytes
            buf = BytesIO()
            image.save(buf, format='PNG')
            img_bytes = buf.getvalue()
            img_path = "temp_mri.png"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            pdf.image(img_path, x=10, y=60, w=100)
            return pdf.output(dest='S').encode('latin1')

        if st.button("Generate PDF Report"):
            pdf_bytes = create_pdf(image, predictions)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="medaix_report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

# --- DASHBOARD PAGE ---
elif page == "Dashboard":
    st.title("📊 Investor Dashboard")
    st.markdown("**Demo usage metrics and AI performance**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Scans Processed", "1,234", "+12 today")
    col2.metric("AI Accuracy", "97.2%", "↑ 0.3%")
    col3.metric("Average Report Time", "4.2s", "↓ 0.1s")
    st.markdown("---")
    st.subheader("Recent Scans")
    # Simulated scan records
    scans = pd.DataFrame({
        "Patient ID": [f"PT-{i:03d}" for i in range(1, 8)],
        "Condition": np.random.choice(["Normal", "Tumor", "Stroke", "Degeneration"], 7),
        "Confidence": [f"{np.random.uniform(85, 99):.1f}%" for _ in range(7)],
        "Time": pd.date_range(end=pd.Timestamp.now(), periods=7, freq='-1D')
    })
    st.dataframe(scans)

# --- ABOUT PAGE ---
elif page == "About MEDAIx":
    st.title("About MEDAIx")
    st.markdown("""
    **MEDAIx** leverages state-of-the-art AI models to empower radiology teams and investors with actionable insights from MRI scans.
    - Founded by experts in AI, healthcare, and medical imaging.
    - Our mission: Accelerate diagnosis, improve outcomes, and enable data-driven investment decisions in medical imaging.
    """)
    st.markdown("**Contact:** invest@medaix.com")
