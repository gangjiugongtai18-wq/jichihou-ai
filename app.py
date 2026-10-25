import streamlit as st
import requests
from openai import OpenAI
import xml.etree.ElementTree as ET

st.set_page_config(
    page_title="自治法AI",
    page_icon="⚖️"
)

st.title("⚖️ 自治法AI")
st.caption("地方自治法に特化したAI法令検索")

st.info(
    "地方自治法の条文を参照してAIが回答します。"
)

@st.cache_data(ttl=86400)
def get_law():
    url = (
        "https://laws.e-gov.go.jp/api/1/lawdata/"
        "322AC0000000067"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.content)

    articles = []
    for article in root.findall(".//Article"):
        num = article.get("Num", "")
        text = "".join(article.itertext())
        articles.append((num, text))

    return articles

question = st.text_area(
    "地方自治法について質問してください",
    placeholder="例：専決処分ができるのはどんな場合？"
)

if st.button("AIに質問する"):
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    try:
        articles = get_law()

        words = question.replace("？", "").split()
        scored = []

        for num, text in articles:
            score = sum(
                text.count(word)
                for word in words
                if len(word) >= 2
            )
            scored.append((score, num, text))

        scored.sort(reverse=True)
        selected = scored[:5]

        if not selected or selected[0][0] == 0:
            st.warning(
                "関連条文を特定できませんでした。"
                "条番号や法律用語を含めて質問してください。"
            )
            st.stop()

        context = "\n\n".join(
            f"第{num}条\n{text[:3000]}"
            for _, num, text in selected
        )

        client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=(
                "あなたは地方自治法の調査補助AIです。"
                "提示された条文だけを根拠に回答してください。"
                "根拠が不足する場合は明示してください。"
                "条番号を捏造しないでください。"
                "回答は結論、根拠条文、解説に分けてください。"
            ),
            input=(
                f"質問：{question}\n\n"
                f"参考条文：\n{context}"
            )
        )

        st.subheader("AIの回答")
        st.write(response.output_text)

        st.subheader("参照した条文")
        for _, num, text in selected:
            with st.expander(f"第{num}条"):
                st.write(text)

        st.markdown(
            "[e-Govで地方自治法を確認]"
            "(https://laws.e-gov.go.jp/law/322AC0000000067)"
        )

    except Exception as e:
        st.error("処理中にエラーが発生しました。")
        st.caption(type(e).__name__)
