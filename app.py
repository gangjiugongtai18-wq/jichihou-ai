import re
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="自治法AI", page_icon="⚖️")

st.title("⚖️ 自治法AI")
st.caption("地方自治法に特化したAI法令検索")
st.info("地方自治法の条文を参照してAIが回答します。")

LAW_ID = "322AC0000000067"
LAW_URL = f"https://laws.e-gov.go.jp/api/1/lawdata/{LAW_ID}"


@st.cache_data(ttl=86400)
def get_law():
    response = requests.get(LAW_URL, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    articles = []

    for article in root.findall(".//Article"):
        num = article.get("Num", "")
        text = "".join(article.itertext()).strip()

        if num and text:
            articles.append((num, text))

    return articles


def find_articles(question, articles):
    match = re.search(r"第\s*(\d+)\s*条", question)

    if not match:
        match = re.fullmatch(r"\s*(\d+)\s*", question)

    if match:
        target = match.group(1)
        return [
            (num, text)
            for num, text in articles
            if num == target
        ]

    keywords = [
        word
        for word in re.split(r"[\s、。？！?]+", question)
        if len(word) >= 2
    ]

    scored = []

    for num, text in articles:
        score = sum(text.count(word) for word in keywords)

        if score > 0:
            scored.append((score, num, text))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        (num, text)
        for _, num, text in scored[:5]
    ]


question = st.text_area(
    "地方自治法について質問してください",
    placeholder="例：地方自治法第179条について教えて"
)

if st.button("AIに質問する"):

    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    try:
        with st.spinner("地方自治法を検索しています..."):
            articles = get_law()
            selected = find_articles(question, articles)

        if not selected:
            st.warning(
                "関連条文を特定できませんでした。"
                "条番号や具体的な法律用語を入力してください。"
            )
            st.stop()

        context = "\n\n".join(
            f"第{num}条\n{text[:3000]}"
            for num, text in selected
        )

        gemini_key = st.secrets.get("GEMINI_API_KEY")

        if not gemini_key:
            st.error(
                "StreamlitのSecretsに"
                "GEMINI_API_KEYを設定してください。"
            )
            st.stop()

        client = OpenAI(
            api_key=gemini_key,
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            )
        )

        with st.spinner("AIが回答を作成しています..."):
            response = client.chat.completions.create(
               model="gemini-3.5-flash-lite", 
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "あなたは地方自治法の調査補助AIです。"
                            "提示された条文を根拠として、"
                            "日本語で分かりやすく回答してください。"
                            "条文にない内容を断定しないでください。"
                            "根拠となる条番号を示してください。"
                            "判断できない場合は"
                            "その旨を明確に伝えてください。"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"質問：{question}\n\n"
                            f"参考条文：\n{context}"
                        )
                    }
                ]
            )

        st.subheader("AIの回答")
        st.write(
            response.choices[0].message.content
            or "回答がありませんでした。"
        )

        st.subheader("参照した条文")

        for num, text in selected:
            with st.expander(f"第{num}条"):
                st.write(text)

        st.markdown(
            "[e-Govで地方自治法を確認]"
            "(https://laws.e-gov.go.jp/law/"
            "322AC0000000067)"
        )

        st.caption(
            "この回答はAIによる参考情報です。"
            "重要な判断では最新の法令と"
            "専門家の確認が必要です。"
        )

    except Exception as e:
        st.error("処理中にエラーが発生しました。")
        st.caption(type(e).__name__)
