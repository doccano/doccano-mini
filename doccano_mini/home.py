from pathlib import Path

import streamlit as st


def main():
    st.set_page_config(page_title="doccano-mini", page_icon=":memo:")
    filepath = Path(__file__).parent.resolve() / "docs" / "README.md"

    # Development
    if not filepath.exists():
        filepath = Path(__file__).parent.parent.resolve() / "README.md"

    with filepath.open("r", encoding="utf-8") as f:
        st.markdown(f.read(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
