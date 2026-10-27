# ============================================================
# SKINO_AI - PROFESSIONAL SKINCARE ASSISTANT
# ============================================================

import streamlit as st
import numpy as np
import json
from pathlib import Path
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="SKINO_AI | Smart Skincare",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- DESIGN ----------------------------

st.markdown(
    """
    <style>
    .stApp { background-color: #F5F8F5; }
    [data-testid="stSidebar"] {
        background-color: #E7F0E8;
        border-right: 1px solid #D8E4D9;
    }
    h1, h2, h3 { color: #24543E; }
    .stButton > button {
        background: #285840;
        color: white;
        border-radius: 10px;
        border: 0;
        font-weight: 600;
    }
    .stButton > button:hover { background: #39765B; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- OPTIONAL IMPORTS --------------------

try:
    from product_database import get_products, get_categories, get_brands
    DATABASE_AVAILABLE = True
    DATABASE_ERROR = ""
except Exception as exc:
    DATABASE_AVAILABLE = False
    DATABASE_ERROR = str(exc)

try:
    from recommendation_engine import get_recommendations
    RECOMMENDATION_AVAILABLE = True
except Exception:
    RECOMMENDATION_AVAILABLE = False

if "predicted_skin_type" not in st.session_state:
    st.session_state["predicted_skin_type"] = None
if "questionnaire_skin_type" not in st.session_state:
    st.session_state["questionnaire_skin_type"] = None

# ------------------------- SIDEBAR ---------------------------

with st.sidebar:
    st.title("🌿 SKINO_AI")
    st.caption("Your Personal Skincare Assistant")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Skin Analysis",
            "Skin Questionnaire",
            "Brand Recommendations",
            "Ingredient Guide",
            "Daily Routine",
            "About & Thank You",
        ],
    )
    st.divider()
    st.caption("AI + Deep Learning Project")
    st.caption("Understand your skin. Care thoughtfully.")

# -------------------- PRODUCT DISPLAY -----------------------

def display_products(products):
    if not products:
        st.info("No matching products found. Try another skin type, category, or brand.")
        return

    st.success(f"{len(products)} catalogue entries found.")
    for start in range(0, len(products), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(products):
                break
            product = products[idx]
            with col:
                with st.container(border=True):
                    st.subheader(f"🧴 {product.get('name', 'Unnamed product')}")
                    st.caption(f"Brand: {product.get('brand', 'Not specified')}")
                    st.write(f"**Category:** {product.get('category', 'Not specified')}")
                    st.write("**Ingredients**")
                    ingredients = product.get("ingredients", [])
                    st.write(", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients))
                    st.write("**Purpose**")
                    st.write(product.get("purpose", "Check the product label."))
                    st.write("**How to use**")
                    st.write(product.get("usage", "Follow the product label."))
                    st.write("**Precautions**")
                    st.write(product.get("precautions", "Check the complete product label before use."))
                    st.caption("Verify current ingredients and product details on the official label before purchasing.")

# --------------------------- HOME ----------------------------

if page == "Home":
    hour = datetime.now().hour
    greeting = "Good Morning" if hour < 12 else "Good Afternoon" if hour < 17 else "Good Evening"

    st.caption("YOUR PERSONAL SKINCARE COMPANION")
    st.title(f"{greeting} 🌿")
    st.header("Welcome to SKINO_AI")
    st.write(
        "A smarter way to explore skincare. Understand ingredients, "
        "discover products, and create a routine that suits your needs."
    )
    st.success("Your skincare journey starts here. Choose a section from the menu to get started.")
    st.divider()
    st.subheader("Explore Your Skincare Journey")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.subheader("📸 AI Skin Analysis")
            st.write("Capture a face image and explore an experimental skin-type prediction.")
    with c2:
        with st.container(border=True):
            st.subheader("🛍️ Product Discovery")
            st.write("Explore products by skin type, category, and brand.")
    with c3:
        with st.container(border=True):
            st.subheader("🧪 Ingredient Guide")
            st.write("Learn about hydration, exfoliating acids, barrier ingredients, and UV filters.")

    st.divider()
    st.subheader("What would you like to explore?")
    left, right = st.columns(2)
    with left:
        st.info("**Personalized guidance**\n\nUse the questionnaire to explore general skincare advice.")
        st.info("**Ingredient education**\n\nUnderstand common skincare ingredients and their uses.")
    with right:
        st.info("**Brand recommendations**\n\nBrowse products with brand and category filters.")
        st.info("**Daily skincare routine**\n\nExplore simple morning and evening steps.")
    st.warning(
        "AI predictions are experimental and are not medical diagnoses. "
        "The current model was trained on a very small dataset."
    )

# ---------------------- SKIN ANALYSIS ------------------------

elif page == "Skin Analysis":
    st.title("📸 AI Skin Analysis")
    st.write("Capture an image using your webcam or upload a photograph.")
    st.warning(
        "This experimental model may make mistakes and cannot diagnose skin conditions. "
        "A single photo cannot reliably determine every person's skin type."
    )

    source = st.radio(
        "Choose image source",
        ["Webcam", "Upload Image"],
        horizontal=True,
    )

    image_file = (
        st.camera_input("Capture your face image")
        if source == "Webcam"
        else st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png"],
        )
    )

    if image_file is not None:
        try:
            pil_image = Image.open(image_file).convert("RGB")
            st.image(
                pil_image,
                caption="Image selected for analysis",
                use_container_width=True,
            )

            if st.button("Analyze Skin Type"):
                import tensorflow as tf

                # Resolve paths relative to this app.py, not a hard-coded PC path.
                project_dir = Path(__file__).resolve().parent
                model_path = project_dir / "models" / "skin_type_mobilenetv2.keras"
                class_path = project_dir / "models" / "skin_type_classes.json"

                if not model_path.exists():
                    st.error(f"Trained model not found: {model_path}")
                    st.stop()

                @st.cache_resource
                def load_model(path):
                    return tf.keras.models.load_model(path)

                model = load_model(str(model_path))

                # Use the exact class order stored during training.
                if class_path.exists():
                    with open(class_path, "r", encoding="utf-8") as file:
                        class_config = json.load(file)
                    class_names = class_config.get(
                        "classes",
                        ["combination", "dry", "normal", "oily"],
                    )
                    image_size = tuple(class_config.get("image_size", [224, 224]))
                else:
                    class_names = ["combination", "dry", "normal", "oily"]
                    image_size = (224, 224)
                    st.warning(
                        "Class mapping file was not found; using the model's expected default class order."
                    )

                if len(image_size) != 2:
                    image_size = (224, 224)

                expected_shape = model.input_shape
                expected_height = expected_shape[1] or image_size[0]
                expected_width = expected_shape[2] or image_size[1]

                resized = pil_image.resize(
                    (int(expected_width), int(expected_height)),
                    Image.Resampling.BILINEAR,
                )
                image_array = np.asarray(resized, dtype=np.float32)

                # IMPORTANT: The trained model already contains its Rescaling
                # preprocessing layer. Pass RGB pixels in the 0-255 range;
                # do not normalize to [-1, 1] here a second time.
                image_array = np.expand_dims(image_array, axis=0)

                scores = model.predict(image_array, verbose=0)[0]

                if len(scores) != len(class_names):
                    st.error(
                        "The model output count does not match the saved class mapping. "
                        "Check skin_type_classes.json and retraining configuration."
                    )
                else:
                    best_idx = int(np.argmax(scores))
                    skin_type = str(class_names[best_idx]).strip().title()
                    st.session_state["predicted_skin_type"] = skin_type

                    st.divider()
                    st.subheader("Your Analysis Results")
                    st.success(f"Model prediction: {skin_type}")
                    st.metric(
                        "Highest model score",
                        f"{float(scores[best_idx]) * 100:.1f}%",
                    )
                    st.caption(
                        "This score is the model's confidence-like output, not real-world accuracy "
                        "or a medically validated probability."
                    )

                    st.subheader("Scores for all skin types")
                    score_cols = st.columns(len(class_names))
                    for i, class_name in enumerate(class_names):
                        value = float(scores[i]) * 100
                        with score_cols[i]:
                            st.metric(str(class_name).title(), f"{value:.1f}%")
                            st.progress(min(max(value / 100, 0.0), 1.0))

                    if RECOMMENDATION_AVAILABLE:
                        try:
                            recommendations = get_recommendations(skin_type)
                            if recommendations:
                                st.subheader("General Ingredient Guidance")
                                for category_name, details in recommendations.items():
                                    with st.expander(str(category_name)):
                                        ingredients = details.get("Ingredients", [])
                                        st.write(
                                            "**Ingredients:**",
                                            ", ".join(ingredients)
                                            if isinstance(ingredients, list)
                                            else str(ingredients),
                                        )
                                        st.write(
                                            "**Purpose:**",
                                            details.get("Benefits", "See product label."),
                                        )
                                        selection = details.get(
                                            "Look For",
                                            details.get(
                                                "Product selection",
                                                "Choose a suitable product.",
                                            ),
                                        )
                                        st.write("**Product selection:**", selection)
                        except Exception as exc:
                            st.info(
                                "The skin-type prediction completed, but ingredient guidance "
                                "could not be loaded."
                            )
                            st.caption(str(exc))

        except Exception as exc:
            st.error(
                "Unable to analyze the image. Check the model file, class mapping, "
                "and TensorFlow installation."
            )
            st.code(str(exc))


# ------------------- SKIN QUESTIONNAIRE ---------------------

elif page == "Skin Questionnaire":
    st.title("📋 Your Skincare Questionnaire")
    st.write("Answer a few questions to explore general skincare guidance.")

    options = ["Not sure", "Dry", "Oily", "Normal", "Combination"]
    prior = st.session_state.get("predicted_skin_type")
    if prior not in options:
        prior = "Not sure"

    skin_type = st.selectbox("How would you describe your skin?", options, index=options.index(prior))
    acne = st.selectbox("How often do you experience acne?", ["Never", "Sometimes", "Frequently"])
    sensitivity = st.selectbox("How sensitive is your skin?", ["Not sensitive", "Sometimes irritated", "Very sensitive"])
    dryness = st.selectbox("How often does your skin feel dry?", ["Never", "Sometimes", "Frequently"])
    pigmentation = st.selectbox("Do you have dark spots or uneven skin tone?", ["No", "Mild", "Noticeable"])

    if st.button("Generate My Guidance"):
        st.session_state["questionnaire_skin_type"] = skin_type
        st.subheader("Your Guidance")

        if skin_type == "Not sure":
            st.info("Your skin type is not confirmed. Start with gentle cleansing, moisturizer, and sunscreen.")
        else:
            st.success(f"Selected skin type: {skin_type}")
            if RECOMMENDATION_AVAILABLE:
                try:
                    recs = get_recommendations(skin_type)
                    for category_name, details in (recs or {}).items():
                        with st.expander(str(category_name)):
                            ingredients = details.get("Ingredients", [])
                            st.write("**Ingredients:**", ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients))
                            st.write(details.get("Benefits", "" ))
                except Exception as exc:
                    st.caption(f"Recommendation details unavailable: {exc}")

        if acne != "Never":
            st.write("**Acne:** Gentle, non-comedogenic products may be helpful. Persistent acne may need a dermatologist's advice.")
        if sensitivity != "Not sensitive":
            st.write("**Sensitivity:** Prefer fragrance-free products and introduce new products one at a time.")
        if dryness != "Never":
            st.write("**Dryness:** Look for moisturizers with ingredients such as glycerin or ceramides.")
        if pigmentation != "No":
            st.write("**Uneven skin tone:** Daily broad-spectrum sunscreen is an important first step.")

        st.subheader("A Simple Daily Routine")
        st.markdown("**Morning:** Gentle cleanser → Moisturizer → Sunscreen\n\n**Night:** Gentle cleanser → Moisturizer")
        st.caption("This is general educational guidance, not a medical diagnosis.")

# -------------------- BRAND RECOMMENDATIONS -----------------

elif page == "Brand Recommendations":
    st.title("🛍️ Skincare Product Discovery")
    st.write("Explore catalogue products by skin type, category, and brand.")

    if not DATABASE_AVAILABLE:
        st.error("Could not import product_database.py.")
        st.code(DATABASE_ERROR)
        st.info("Fix any syntax errors in product_database.py, then rerun the app.")
    else:
        skin_options = ["Oily", "Dry", "Normal", "Combination"]
        prior = st.session_state.get("questionnaire_skin_type") or st.session_state.get("predicted_skin_type")
        if prior not in skin_options:
            prior = "Oily"

        selected_skin = st.selectbox("Select skin type", skin_options, index=skin_options.index(prior))
        try:
            categories = ["All"] + list(get_categories())
            brands = ["All"] + list(get_brands())
            selected_category = st.selectbox("Select category", categories)
            selected_brand = st.selectbox("Select brand", brands)
            products = get_products(selected_skin, selected_category, selected_brand)
            display_products(products)
        except Exception as exc:
            st.error("Could not retrieve products from product_database.py.")
            st.code(str(exc))

# ----------------------- INGREDIENT GUIDE -------------------

elif page == "Ingredient Guide":
    st.title("🧪 Skincare Ingredient Guide")
    st.write("Search by ingredient name, type, purpose, or skincare concern.")

    ingredients = {
        "Salicylic Acid": {
            "type": "BHA",
            "purpose": "May help unclog pores and manage some acne.",
            "used_in": "Some cleansers and acne treatments.",
            "precaution": "May cause dryness or irritation."
        },
        "Glycolic Acid": {
            "type": "AHA",
            "purpose": "Exfoliation and uneven skin texture.",
            "used_in": "Some exfoliating products.",
            "precaution": "Can irritate sensitive skin."
        },
        "Lactic Acid": {
            "type": "AHA",
            "purpose": "Exfoliation and skin texture support.",
            "used_in": "Some exfoliating treatments.",
            "precaution": "Introduce gradually."
        },
        "Hyaluronic Acid": {
            "type": "Humectant",
            "purpose": "Helps attract and retain moisture; hydration.",
            "used_in": "Serums and moisturizers.",
            "precaution": "Follow product instructions."
        },
        "Niacinamide": {
            "type": "Vitamin B3",
            "purpose": "May support the skin barrier and help manage oiliness.",
            "used_in": "Serums and moisturizers.",
            "precaution": "Stop if persistent irritation occurs."
        },
        "Ceramides": {
            "type": "Skin barrier lipids",
            "purpose": "Help support the skin barrier and reduce moisture loss.",
            "used_in": "Moisturizers and cleansers.",
            "precaution": "Check the complete product formulation."
        },
        "Glycerin": {
            "type": "Humectant",
            "purpose": "Helps attract and retain water; hydration.",
            "used_in": "Cleansers and moisturizers.",
            "precaution": "Check the full formulation."
        },
        "Vitamin C": {
            "type": "Antioxidant",
            "purpose": "May help improve the appearance of uneven skin tone and pigmentation.",
            "used_in": "Some serums and moisturizers.",
            "precaution": "Some formulations can irritate sensitive skin."
        },
        "Zinc Oxide": {
            "type": "Mineral UV filter",
            "purpose": "Helps protect skin from UV radiation; sunscreen.",
            "used_in": "Sunscreens.",
            "precaution": "Choose a properly labelled sunscreen."
        },
        "Titanium Dioxide": {
            "type": "Mineral UV filter",
            "purpose": "Helps protect skin from UV radiation; sunscreen.",
            "used_in": "Sunscreens.",
            "precaution": "Check the sunscreen's labelled protection."
        },
        "Panthenol": {
            "type": "Pro-vitamin B5",
            "purpose": "Supports hydration and skin comfort.",
            "used_in": "Moisturizers and serums.",
            "precaution": "Check the complete formulation."
        },
        "Azelaic Acid": {
            "type": "Dicarboxylic acid",
            "purpose": "May help with acne and uneven skin tone.",
            "used_in": "Selected treatment products.",
            "precaution": "May cause irritation; introduce carefully."
        },
        "Retinol": {
            "type": "Retinoid",
            "purpose": "Used in some products for acne and signs of skin aging.",
            "used_in": "Selected night treatments.",
            "precaution": "Can irritate skin. Avoid during pregnancy unless a qualified clinician advises otherwise."
        },
        "Peptides": {
            "type": "Skin-conditioning ingredients",
            "purpose": "Used in formulations targeting skin appearance.",
            "used_in": "Some serums and moisturizers.",
            "precaution": "Benefits depend on the complete formulation."
        },
        "Squalane": {
            "type": "Emollient",
            "purpose": "Helps soften skin and reduce moisture loss.",
            "used_in": "Moisturizers and facial oils.",
            "precaution": "Check the complete formulation."
        }
    }

    query = st.text_input(
        "Search ingredients",
        placeholder="Try hydration, acne, acid, barrier, or sunscreen",
        key="ingredient_search",
    ).strip().casefold()

    matches = []
    for name, details in ingredients.items():
        searchable = " ".join([name] + list(details.values())).casefold()
        if not query or query in searchable:
            matches.append((name, details))

    if matches:
        st.caption(f"{len(matches)} matching ingredients found.")
        for name, details in matches:
            with st.expander(name):
                st.write(f"**Ingredient type:** {details['type']}")
                st.write(f"**Purpose:** {details['purpose']}")
                st.write(f"**Common uses:** {details['used_in']}")
                st.write(f"**Precautions:** {details['precaution']}")
    else:
        st.warning("No matching ingredients found. Try another keyword.")

    st.info("Sunscreen does not need an acid. Prioritize suitable UV filters and labelled broad-spectrum SPF protection.")

# ----------------------- DAILY ROUTINE ----------------------

elif page == "Daily Routine":
    st.title("🌞 Your Daily Skincare Routine")
    morning, evening = st.columns(2)

    with morning:
        with st.container(border=True):
            st.subheader("☀️ Morning")
            st.markdown(
                """
                **Step 1 — Cleanse**  
                Use a gentle cleanser if needed.

                **Step 2 — Moisturize**  
                Apply a suitable moisturizer.

                **Step 3 — Protect**  
                Use broad-spectrum SPF 30+ sunscreen. Reapply as directed on the label.
                """
            )

    with evening:
        with st.container(border=True):
            st.subheader("🌙 Evening")
            st.markdown(
                """
                **Step 1 — Cleanse**  
                Remove dirt, oil, and sunscreen.

                **Step 2 — Optional treatment**  
                Use a suitable treatment only if needed.

                **Step 3 — Moisturize**  
                Support hydration and the skin barrier.
                """
            )

    st.divider()
    st.subheader("Important Tips")
    st.markdown(
        """
        - Introduce one new active ingredient at a time.
        - Avoid excessive exfoliation.
        - Patch-test new products.
        - Stop products that cause persistent irritation.
        - Use sunscreen regularly.
        - Consult a dermatologist for persistent skin concerns.
        """
    )

# -------------------- ABOUT & THANK YOU ---------------------

elif page == "About & Thank You":
    st.title("💚 About SKINO_AI")
    st.subheader("Skincare, Made Simpler.")
    st.write(
        "SKINO_AI is an AI and deep-learning project designed to make "
        "skincare information easier to explore and understand."
    )

    st.divider()
    st.subheader("What Our Application Offers")
    st.markdown(
        """
        - 📸 Experimental image-based skin-type prediction
        - 🛍️ Product discovery with brand and category filters
        - 🧪 Skincare ingredient education
        - 📋 A skin questionnaire
        - 🌞 Morning and evening routine guidance
        """
    )

    st.warning(
        "SKINO_AI is an educational project. Its model has a small "
        "training dataset and must not be used to diagnose skin conditions "
        "or replace professional medical advice."
    )

    st.divider()
    st.success("🌿 THANK YOU FOR VISITING SKINO_AI")
    st.header("Your Skincare Journey Matters 💚")
    st.write(
        "Thank you for exploring our skincare assistant. We hope it helps "
        "you learn about ingredients, routines, and product choices."
    )
    st.info("Understand your skin. Choose thoughtfully. Care consistently.")
    st.markdown("### Made with care as an AI & Deep Learning project.")
    st.caption("Thank you for being part of the SKINO_AI journey.")

# --------------------------- FOOTER -------------------------

st.divider()
left, right = st.columns(2)
with left:
    st.caption("🌿 SKINO_AI | Your Personal Skincare Assistant")
with right:
    st.caption("AI-powered educational project | Not medical advice")
