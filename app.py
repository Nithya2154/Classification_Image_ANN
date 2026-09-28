import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)


# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_model.keras")


model = load_model()


# --------------------------------
# Title
# --------------------------------

st.title("🔢 MNIST Handwritten Digit Classifier")

st.write(
    "Upload a handwritten digit image and the neural network "
    "will predict the digit from 0 to 9."
)


# --------------------------------
# Upload Image
# --------------------------------

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)


if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=200
    )

    # --------------------------------
    # Preprocessing
    # --------------------------------

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to numpy array
    image_array = np.array(image)

    # Convert to float32
    image_array = image_array.astype("float32")

    # Normalize
    image_array = image_array / 255.0

    # Flatten
    image_array = image_array.reshape(1, 28 * 28)

    # --------------------------------
    # Prediction
    # --------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # --------------------------------
    # Display Result
    # --------------------------------

    st.success(
        f"Predicted Digit: {predicted_digit}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )

    # --------------------------------
    # Probability
    # --------------------------------

    st.subheader("Prediction Probabilities")

    probabilities = prediction[0]

    for digit, probability in enumerate(probabilities):

        st.write(
            f"Digit {digit}: {probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )
