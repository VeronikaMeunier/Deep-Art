import streamlit as st

st.image(
    "https://cdn.artificialpaintings.com/uploads/uploads/2024/09/ai-3-768x439.webp",
    use_container_width=True)

st.title("Welcome to Deep Art!")

st.markdown("""
    ****Deep Art**** is a visual AI app designed to explore the world of fine art through the lens of deep learning. This app features two key components - Author Classification and Style Transfer.
""")

if 'total_count' not in st.session_state:
    st.session_state['total_count'] = 0

if 'total_count_stylized' not in st.session_state:
    st.session_state['total_count_stylized'] = 0

col1, col2 = st.columns(2)

with col1:
    st.header("🧑‍🎨 Author Classification", divider='rainbow')

    st.markdown("""
    <p style="text-align: justify;">
    Discover the likely creator of a given piece of art using a custom trained deep learning model. By analyzing patterns, colors, and textures, the model predicts the most probable artist behind a work. 
    </p>
    """,
                unsafe_allow_html=True)

    st.page_link("pages/1_🧑‍🎨_Author_Classification.py",
                 label=":orange[**Try it now!**]",
                 use_container_width=True)

    st.metric("Classified images", st.session_state['total_count'])

with col2:
    st.header("🖌️ Neural Style Transfer", divider='rainbow')

    st.markdown("""
    <p style="text-align: justify;">
    Reimagine your images with the visual style of famous paintings! NST allows you to blend the content of one image with the style of another, creating beautiful and artistic transformations.
    </p>         
    """,
                unsafe_allow_html=True)

    st.page_link("pages/2_🖌️_Neural_Style_Transfer.py",
                 label=":orange[**Try it now!**]",
                 use_container_width=True)

    st.metric("Stylized images", st.session_state['total_count_stylized'])
