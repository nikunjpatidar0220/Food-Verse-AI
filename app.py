import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Food Verse AI", page_icon="🍎", layout="wide")

st.title("🍎 Food Verse AI")
st.write("Your Multimodal Food Intelligence Assistant")
st.divider()


@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline("image-classification", model="nateraw/food")


st.header("AI Food Recognition")

uploaded_file = st.file_uploader("Choose a food image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded food image", use_container_width=True)

    if st.button("🔍 Recognize Food"):
        try:
            with st.spinner(
                "Loading the AI model for the first time. "
                "This may take several minutes..."
            ):
                classifier = load_model()

            with st.spinner("Analyzing your image..."):
                results = classifier(Image.open(uploaded_file).convert("RGB"))

            st.subheader("AI Predictions")

            for result in results[:5]:
                st.write(f"**{result['label']}** — " f"{result['score'] * 100:.2f}%")
                st.caption(
                    "Scores are model confidence estimates, "
                    "not guarantees of accuracy."
                )

            # Food information database

            food_database = {
                "samosa": {
                    "description": "A crispy Indian snack commonly filled with spiced potatoes and peas.",
                    "calories": 260,
                    "protein": 5,
                    "carbs": 30,
                    "fat": 13,
                },
                "dosa": {
                    "description": "A thin South Indian crepe usually made from fermented rice and lentil batter.",
                    "calories": 168,
                    "protein": 4,
                    "carbs": 29,
                    "fat": 4,
                },
                "idli": {
                    "description": "A soft, steamed South Indian rice and lentil cake.",
                    "calories": 58,
                    "protein": 2,
                    "carbs": 12,
                    "fat": 0.4,
                },
                "pizza": {
                    "description": "A baked dish made with a base, sauce, cheese and optional toppings.",
                    "calories": 285,
                    "protein": 12,
                    "carbs": 36,
                    "fat": 10,
                },
                "hamburger": {
                    "description": "A sandwich with a bun and a patty, often served with vegetables and sauces.",
                    "calories": 295,
                    "protein": 17,
                    "carbs": 24,
                    "fat": 14,
                },
                "biryani": {
                    "description": "A spiced rice dish prepared with vegetables or meat and aromatic seasonings.",
                    "calories": 300,
                    "protein": 10,
                    "carbs": 40,
                    "fat": 10,
                },
            }

            # Get the highest-scoring prediction
            top_food = results[0]["label"].lower().replace("_", " ")
            st.divider()
            st.header("🍽️ Food Information")

            if top_food in food_database:
                info = food_database[top_food]

                st.subheader(top_food.title())
                st.write(info["description"])

                st.write("Estimated nutrition per medium samosa:")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Calories", f"{info['calories']} kcal")
                    st.metric("Protein", f"{info['protein']} g")

                with col2:
                    st.metric("Carbohydrates", f"{info['carbs']} g")
                    st.metric("Fat", f"{info['fat']} g")

                st.caption(
                    "Nutrition values are approximate estimates. "
                    "Actual values depend on size, ingredients and cooking method."
                )

            else:
                st.info(
                    f"We recognized '{top_food}', but food information "
                    "for this item has not been added yet."
                )

        except Exception as error:
            st.error("The model could not finish loading or analyzing.")
            st.code(str(error))
            st.info(
                "Check your internet connection and the VS Code "
                "terminal for additional details."
            )
