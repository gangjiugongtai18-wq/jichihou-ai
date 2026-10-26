import re
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from openai import OpenAI


# =========================================================
# Streamlit 基本設定
# =========================================================

st.set_page_config(
    page_title="自治法AI",
    page_icon="⚖️"
)

st.title("⚖️ 自治法AI")
st.caption("地方自治法に特化したAI法令検索")
st.info("地方自治法の条文を参照してAIが回答します。")


# =========================================================
# 地方自治法
# =========================================================

LAW_ID = "322AC0000000067"
LAW_URL = f"https://laws.e-gov.go.jp/api/1/lawdata/{LAW_ID}"


# =========================================================
# e-Govから地方自治法を取得
# =========================================================

@st.cache_data(ttl=86400)
def get_law():
    response = requests.get(
        LAW_URL,
        timeout=30
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)

    articles = []

    for article in root.findall(".//Article"):
        num = article.get("Num", "")
        text = "".join(article.itertext()).strip()

        if num and text:
            articles.append((num, text))

    return articles


# =========================================================
# Geminiクライアント
# =========================================================

def get_ai_client():

    gemini_key = st.secrets.get("GEMINI_API_KEY")

    if not gemini_key:
        st.error(
            "StreamlitのSecretsに"
            "GEMINI_API_KEYを設定してください。"
        )
        st.stop()

    return OpenAI(
        api_key=gemini_key,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        )
    )


# =========================================================
# Geminiに検索キーワードを作らせる
# =========================================================

def make_search_keywords(question, client):

    response = client.chat.completions.create(
        model="gemini-3.5-flash-lite",
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは地方自治法の条文検索を補助するAIです。"
                    "利用者の質問から、関係する地方自治法の条文を"
                    "検索するためのキーワードを抽出してください。"

                    "質問文に直接書かれていなくても、"
                    "質問の意味から関係する可能性が高い"
                    "地方自治法上の法律用語を補ってください。"

                    "特に制度名、法律用語、議会、長、"
                    "行政機関などに関係する重要語を優先してください。"

                    "5個から10個程度のキーワードを出してください。"

                    "回答はキーワードのみを"
                    "読点区切りで返してください。"

                    "説明文は書かないでください。"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    result = (
        response.choices[0].message.content
        or ""
    )

    keywords = [
        word.strip()
        for word in re.split(
            r"[、,\n]+",
            result
        )
        if len(word.strip()) >= 2
    ]

    return keywords


# =========================================================
# 条文検索
# =========================================================

def find_articles(question, articles, client):

    # -----------------------------------------------------
    # 1. 条番号が明示されている場合
    # -----------------------------------------------------

    match = re.search(
        r"第\s*(\d+)\s*条",
        question
    )

    if not match:
        match = re.fullmatch(
            r"\s*(\d+)\s*",
            question
        )

    if match:

        target = match.group(1)

        return [
            (num, text)
            for num, text in articles
            if num == target
        ], []


    # -----------------------------------------------------
    # 2. 自然文の場合
    # Geminiに検索語を作らせる
    # -----------------------------------------------------

    keywords = make_search_keywords(
        question,
        client
    )

    scored = []

    for num, text in articles:

        score = 0

        for word in keywords:

            count = text.count(word)

            if count > 0:
                score += count

                # 長い専門用語ほど少し高く評価
                if len(word) >= 4:
                    score += 2

        if score > 0:
            scored.append(
                (
                    score,
                    num,
                    text
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = [
        (num, text)
        for _, num, text
        in scored[:5]
    ]

    return selected, keywords


# =========================================================
# 質問入力
# =========================================================

question = st.text_area(
    "地方自治法について質問してください",
    placeholder=(
        "例：議会を招集する時間がないとき、"
        "首長はどうすればいい？"
    )
)


# =========================================================
# 質問ボタン
# =========================================================

if st.button("AIに質問する"):

    if not question.strip():

        st.warning(
            "質問を入力してください。"
        )

        st.stop()


    try:

        # -------------------------------------------------
        # Geminiクライアント
        # -------------------------------------------------

        client = get_ai_client()


        # -------------------------------------------------
        # 地方自治法取得
        # -------------------------------------------------

        with st.spinner(
            "地方自治法を検索しています..."
        ):

            articles = get_law()

            selected, keywords = find_articles(
                question,
                articles,
                client
            )


        # -------------------------------------------------
        # 検索語を表示
        # -------------------------------------------------

        if keywords:

            with st.expander(
                "AIが使用した検索キーワード"
            ):

                st.write(
                    " / ".join(keywords)
                )


        # -------------------------------------------------
        # 条文が見つからない場合
        # -------------------------------------------------

        if not selected:

            st.warning(
                "関連条文を特定できませんでした。"
                "質問の表現を変えるか、"
                "条番号や具体的な法律用語を"
                "入力してください。"
            )

            st.stop()


        # -------------------------------------------------
        # Geminiに渡す条文
        # -------------------------------------------------

        context = "\n\n".join(
            f"第{num}条\n{text[:3000]}"
            for num, text in selected
        )


        # -------------------------------------------------
        # AI回答生成
        # -------------------------------------------------

        with st.spinner(
            "AIが回答を作成しています..."
        ):

            response = (
                client.chat.completions.create(
                    model="gemini-3.5-flash-lite",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "あなたは地方自治法の"
                                "調査補助AIです。"

                                "提示された条文だけを"
                                "主要な根拠として、"
                                "利用者の質問に"
                                "日本語で分かりやすく"
                                "回答してください。"

                                "まず質問に対する結論を"
                                "簡潔に示してください。"

                                "その後、根拠となる条文と"
                                "理由を説明してください。"

                                "提示された複数の条文のうち、"
                                "質問と直接関係する条文を"
                                "優先してください。"

                                "条文に書かれていない内容を"
                                "条文に書かれているかのように"
                                "断定しないでください。"

                                "各説明について、可能な限り"
                                "根拠となる条番号・項番号を"
                                "示してください。"

                                "提示された条文だけでは"
                                "判断できない場合は、"
                                "その旨を明確にしてください。"
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
            )


        # -------------------------------------------------
        # AI回答表示
        # -------------------------------------------------

        st.subheader("AIの回答")

        answer = (
            response.choices[0].message.content
            or "回答がありませんでした。"
        )

        st.write(answer)


        # -------------------------------------------------
        # 参照条文
        # -------------------------------------------------

        st.subheader("参照した条文")

        for num, text in selected:

            with st.expander(
                f"第{num}条"
            ):

                st.write(text)


        # -------------------------------------------------
        # e-Govリンク
        # -------------------------------------------------

        st.markdown(
            "[e-Govで地方自治法を確認]"
            "(https://laws.e-gov.go.jp/law/"
            "322AC0000000067)"
        )


        # -------------------------------------------------
        # 注意事項
        # -------------------------------------------------

        st.caption(
            "この回答はAIによる法令調査の"
            "参考情報です。"
            "AIは誤った回答を生成する可能性があります。"
            "重要な判断では最新の法令、"
            "公的資料および専門家の確認が必要です。"
        )


    except Exception as e:

        st.error(
            "処理中にエラーが発生しました。"
        )

        st.caption(
            f"{type(e).__name__}: {e}"
        )
