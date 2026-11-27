import streamlit as st
import os
import tempfile
import io

from PIL import Image

from compression import compress_image
from decompression import decompress_image


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Vector Quantization",
    page_icon="🖼️",
    layout="wide"
)


# ================= SESSION STATE =================

if "compressed_data" not in st.session_state:
    st.session_state.compressed_data = None

if "codebook_data" not in st.session_state:
    st.session_state.codebook_data = None

if "decompressed_data" not in st.session_state:
    st.session_state.decompressed_data = None

if "compression_ratio" not in st.session_state:
    st.session_state.compression_ratio = None


# ================= TITLE =================

st.title("🖼️ Vector Quantization Image Compressor")

st.write(
    "Compress and decompress grayscale images using "
    "the Vector Quantization algorithm."
)

st.divider()


# ================= MAIN FORM =================

option = st.radio(
    "Choose an operation:",
    ["Compression", "Decompression"],
    horizontal=True
)


# ================= COMPRESSION =================

if option == "Compression":

    st.header("Image Compression")

    with st.form("compression_form"):

        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg", "bmp"]
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            block_height = st.number_input(
                "Block Height",
                min_value=1,
                value=2,
                step=1
            )

        with col2:
            block_width = st.number_input(
                "Block Width",
                min_value=1,
                value=2,
                step=1
            )

        with col3:
            codebook_size = st.number_input(
                "Codebook Size",
                min_value=1,
                value=8,
                step=1
            )

        submit_compression = st.form_submit_button(
            "Compress Image"
        )

    if submit_compression:

        if uploaded_image is None:
            st.warning("Please upload an image.")

        else:

            try:

                os.makedirs("Output", exist_ok=True)

                # Save uploaded image temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".png"
                ) as temp_file:

                    temp_file.write(uploaded_image.getvalue())
                    image_path = temp_file.name

                # Show original image
                original_image = Image.open(
                    io.BytesIO(uploaded_image.getvalue())
                ).convert("L")

                st.subheader("Original Image")
                st.image(
                    original_image,
                    caption="Original Image",
                    width=350
                )

                # Run compression
                with st.spinner("Compressing image..."):

                    result = compress_image(
                        image_path,
                        (int(block_height), int(block_width)),
                        int(codebook_size),
                        codebook_path="Output/codebook.npz",
                        compressed_path="Output/compressed.bin"
                    )

                # Read output files
                with open("Output/compressed.bin", "rb") as f:
                    st.session_state.compressed_data = f.read()

                with open("Output/codebook.npz", "rb") as f:
                    st.session_state.codebook_data = f.read()

                st.session_state.decompressed_data = None

                # Calculate compression ratio
                from codebook import load_codebook

                info = load_codebook("Output/codebook.npz")

                original_size = (
                    info["original_shape"][0]
                    * info["original_shape"][1]
                    * 8
                )

                compressed_size = (
                    len(st.session_state.compressed_data) * 8
                    + len(st.session_state.codebook_data) * 8
                )

                ratio = original_size / compressed_size

                st.session_state.compression_ratio = ratio

                st.success("Image compressed successfully!")

                st.subheader("Compression Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Compression Ratio",
                        f"{ratio:.2f}:1"
                    )

                with col2:
                    st.metric(
                        "Original Size",
                        f"{original_size / 8:.2f} Bytes"
                    )

                with col3:
                    st.metric(
                        "Compressed Size",
                        f"{compressed_size / 8:.2f} Bytes"
                    )

                st.info(
                    "The Codebook and Binary Stream are ready "
                    "for download."
                )

                os.remove(image_path)

            except Exception as e:
                st.error(f"Compression failed: {e}")


    # Download buttons

    if st.session_state.compressed_data is not None:

        st.subheader("Download Files")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Codebook",
                data=st.session_state.codebook_data,
                file_name="codebook.npz",
                mime="application/octet-stream"
            )

        with col2:
            st.download_button(
                "Download Compressed Image",
                data=st.session_state.compressed_data,
                file_name="compressed.bin",
                mime="application/octet-stream"
            )


# ================= DECOMPRESSION =================

elif option == "Decompression":

    st.header("Image Decompression")

    with st.form("decompression_form"):

        uploaded_codebook = st.file_uploader(
            "Upload Codebook File",
            type=["npz"]
        )

        uploaded_compressed = st.file_uploader(
            "Upload Compressed Binary File",
            type=["bin"]
        )

        submit_decompression = st.form_submit_button(
            "Decompress Image"
        )

    if submit_decompression:

        if uploaded_codebook is None or uploaded_compressed is None:

            st.warning(
                "Please upload both the Codebook and "
                "Compressed Binary files."
            )

        else:

            try:

                os.makedirs("Output", exist_ok=True)

                # Save uploaded files temporarily
                with open("Output/codebook_input.npz", "wb") as f:
                    f.write(uploaded_codebook.getvalue())

                with open("Output/compressed_input.bin", "wb") as f:
                    f.write(uploaded_compressed.getvalue())

                # Run decompression
                with st.spinner("Decompressing image..."):

                    decompress_image(
                        codebook_path="Output/codebook_input.npz",
                        compressed_path="Output/compressed_input.bin",
                        output_path="Output/decompressed.png"
                    )

                # Read decompressed image
                with open("Output/decompressed.png", "rb") as f:
                    st.session_state.decompressed_data = f.read()

                st.success(
                    "Image decompressed successfully!"
                )

                st.subheader("Reconstructed Image")

                st.image(
                    st.session_state.decompressed_data,
                    caption="Decompressed Image",
                    width=400
                )

            except Exception as e:

                st.error(
                    f"Decompression failed: {e}"
                )


    # Download decompressed image

    if st.session_state.decompressed_data is not None:

        st.download_button(
            "Download Decompressed Image",
            data=st.session_state.decompressed_data,
            file_name="decompressed.png",
            mime="image/png"
        )
