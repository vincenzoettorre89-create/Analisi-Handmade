import streamlit as st
from PIL import Image
import material_price_estimator as analyzer
import tempfile

st.set_page_config(page_title="Analisi Handmade", layout="centered")

st.title("📦 Analisi Prodotti Handmade")
st.write("Carica una foto del prodotto e ottieni stima materiale e prezzo.")

uploaded_file = st.file_uploader("Carica immagine", type=["jpg", "jpeg", "png"])

real_width = st.number_input("Larghezza reale del prodotto (cm)", min_value=1.0, step=0.5)

if uploaded_file and real_width:

    image = Image.open(uploaded_file)
    st.image(image, caption="Immagine caricata", use_column_width=True)

    with st.spinner("Analisi in corso..."):
        # Salviamo temporaneamente l'immagine
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            result = analyzer.analyze_image(tmp.name, real_width_cm=real_width)

    st.success("Analisi completata!")

    materials = result["material_scores"]
    prices = result["price_info"]["price_suggestion"]
    costs = result["price_info"]["costs"]

    st.subheader("Materiali stimati")
    for m, s in list(materials.items())[:5]:
        st.write(f"**{m}**: {round(s*100,1)}%")

    st.subheader("Costi")
    st.write(f"Materiale: €{round(costs['material_cost'],2)}")
    st.write(f"Lavoro: €{round(costs['labor_cost'],2)}")
    st.write(f"Base: €{round(costs['base_cost'],2)}")

    st.subheader("Prezzo suggerito")
    st.write(f"💰 Basso: €{round(prices['low'],2)}")
    st.write(f"💰 Tipico: €{round(prices['typical'],2)}")
    st.write(f"💰 Alto: €{round(prices['high'],2)}")
