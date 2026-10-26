import json
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
st.info(
    "質問の意味をAIが分析し、"
    "e-Govから取得した地方自治法の条文を参照して回答します。"
)


# =========================================================
# 地方自治法
# =========================================================

LAW_ID = "322AC0000000067"
LAW_URL = f"https://laws.e-gov.go.jp/api/1/lawdata/{LAW_ID}"
EGOV_URL = f"https://laws.e-gov.go.jp/law/{LAW_ID}"

MODEL_NAME = "gemini-3.5-flash-lite"


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

    root = ET.fromstring(
        response.content
    )

    articles = []

    for article in root.findall(".//Article"):

        num = article.get(
            "Num",
            ""
        ).strip()

        text = "".join(
            article.itertext()
        ).strip()

        if num and text:

            articles.append(
                (num, text)
            )

    return articles


# =========================================================
# Geminiクライアント
# =========================================================

def get_ai_client():

    gemini_key = st.secrets.get(
        "GEMINI_API_KEY"
    )

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
# JSON文字列を安全に取り出す
# =========================================================

def extract_json(text):

    if not text:
        return None

    text = text.strip()

    # ```json ～ ``` を除去
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # JSON部分だけを探す
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return None

    try:
        return json.loads(
            match.group(0)
        )

    except json.JSONDecodeError:
        return None


# =========================================================
# 条番号を正規化
# 例：
# "179" → "179"
# "第179条" → "179"
# =========================================================

def normalize_article_number(value):

    if value is None:
        return None

    value = str(value).strip()

    match = re.search(
        r"(\d+)",
        value
    )

    if not match:
        return None

    return match.group(1)


# =========================================================
# 質問をGeminiに分析させる
# =========================================================

def analyze_question(
    question,
    client
):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは日本の地方自治法の"
                    "条文検索を補助するAIです。"

                    "利用者の質問を分析し、"
                    "地方自治法上の法的論点、"
                    "関係する可能性が高い条番号、"
                    "条文検索に使う法律用語を"
                    "推定してください。"

                    "日常語を法律用語に"
                    "言い換えることが重要です。"

                    "例えば、"
                    "『市長が辞めた』という表現から、"
                    "『長が欠けた場合』"
                    "『職務代理』などの"
                    "法律用語を検討してください。"

                    "ただし、分からない条番号を"
                    "無理に作らないでください。"

                    "候補条文は最大5件にしてください。"

                    "必ず次のJSON形式だけで"
                    "回答してください。"

                    "{"
                    "\"issue\":\"法的論点\","
                    "\"candidate_articles\":[\"179\",\"180\"],"
                    "\"keywords\":["
                    "\"専決処分\","
                    "\"議会\","
                    "\"長\""
                    "]"
                    "}"

                    "JSON以外の説明文は"
                    "絶対に書かないでください。"
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    result = (
        response.choices[0]
        .message.content
        or ""
    )

    data = extract_json(
        result
    )

    if not data:

        return {
            "issue": "",
            "candidate_articles": [],
            "keywords": []
        }

    issue = str(
        data.get(
            "issue",
            ""
        )
    ).strip()

    raw_articles = data.get(
        "candidate_articles",
        []
    )

    raw_keywords = data.get(
        "keywords",
        []
    )

    if not isinstance(
        raw_articles,
        list
    ):
        raw_articles = []

    if not isinstance(
        raw_keywords,
        list
    ):
        raw_keywords = []

    candidate_articles = []

    for value in raw_articles:

        num = normalize_article_number(
            value
        )

        if (
            num
            and num not in candidate_articles
        ):
            candidate_articles.append(
                num
            )

    keywords = []

    for word in raw_keywords:

        word = str(
            word
        ).strip()

        if (
            len(word) >= 2
            and word not in keywords
        ):
            keywords.append(
                word
            )

    return {
        "issue": issue,
        "candidate_articles":
            candidate_articles[:5],
        "keywords":
            keywords[:10]
    }


# =========================================================
# 質問に条番号が直接書かれているか
# =========================================================

def get_direct_article_number(
    question
):

    match = re.search(
        r"第\s*(\d+)\s*条",
        question
    )

    if match:
        return match.group(1)

    match = re.fullmatch(
        r"\s*(\d+)\s*",
        question
    )

    if match:
        return match.group(1)

    return None


# =========================================================
# 条番号から実際の条文を取得
# =========================================================

def get_articles_by_numbers(
    numbers,
    articles
):

    wanted = set(
        numbers
    )

    result = []

    for num, text in articles:

        if num in wanted:

            result.append(
                (num, text)
            )

    return result


# =========================================================
# キーワードによる補助検索
# =========================================================

def keyword_search(
    keywords,
    articles,
    limit=5
):

    scored = []

    for num, text in articles:

        score = 0
        matched_words = 0

        for word in keywords:

            count = text.count(
                word
            )

            if count > 0:

                matched_words += 1

                # 基本点
                score += count

                # 長い法律用語を高く評価
                if len(word) >= 4:
                    score += 3

                elif len(word) >= 3:
                    score += 2

        # 複数の検索語が同じ条文に
        # 含まれる場合は追加点
        if matched_words >= 2:
            score += (
                matched_words * 2
            )

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

    return [
        (num, text)
        for score, num, text
        in scored[:limit]
    ]


# =========================================================
# 重複条文を除去
# =========================================================

def merge_articles(
    primary,
    secondary,
    limit=5
):

    merged = []
    seen = set()

    for num, text in (
        primary + secondary
    ):

        if num in seen:
            continue

        seen.add(
            num
        )

        merged.append(
            (num, text)
        )

        if len(merged) >= limit:
            break

    return merged


# =========================================================
# 関連条文を検索
# =========================================================

def find_articles(
    question,
    articles,
    client
):

    # -----------------------------------------------------
    # 1. 条番号が直接指定された場合
    # -----------------------------------------------------

    direct_number = (
        get_direct_article_number(
            question
        )
    )

    if direct_number:

        selected = (
            get_articles_by_numbers(
                [direct_number],
                articles
            )
        )

        analysis = {
            "issue":
                "利用者が条番号を直接指定",
            "candidate_articles":
                [direct_number],
            "keywords":
                []
        }

        return (
            selected,
            analysis
        )


    # -----------------------------------------------------
    # 2. Geminiが質問の法的論点を分析
    # -----------------------------------------------------

    analysis = analyze_question(
        question,
        client
    )

    candidate_numbers = (
        analysis[
            "candidate_articles"
        ]
    )

    keywords = (
        analysis[
            "keywords"
        ]
    )


    # -----------------------------------------------------
    # 3. Geminiが推定した条番号を
    #    e-Govデータから取得
    # -----------------------------------------------------

    candidate_results = (
        get_articles_by_numbers(
            candidate_numbers,
            articles
        )
    )


    # -----------------------------------------------------
    # 4. キーワード検索も実施
    # -----------------------------------------------------

    keyword_results = (
        keyword_search(
            keywords,
            articles,
            limit=5
        )
    )


    # -----------------------------------------------------
    # 5. 候補条番号を優先し、
    #    キーワード検索結果を補完
    # -----------------------------------------------------

    selected = merge_articles(
        candidate_results,
        keyword_results,
        limit=5
    )

    return (
        selected,
        analysis
    )


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

if st.button(
    "AIに質問する"
):

    if not question.strip():

        st.warning(
            "質問を入力してください。"
        )

        st.stop()

    try:

        # -------------------------------------------------
        # Gemini
        # -------------------------------------------------

        client = get_ai_client()


        # -------------------------------------------------
        # 地方自治法を取得
        # -------------------------------------------------

        with st.spinner(
            "質問の意味と地方自治法を"
            "調査しています..."
        ):

            articles = get_law()

            selected, analysis = (
                find_articles(
                    question,
                    articles,
                    client
                )
            )


        # -------------------------------------------------
        # 検索分析結果
        # -------------------------------------------------

        with st.expander(
            "AIによる検索分析"
        ):

            issue = analysis.get(
                "issue",
                ""
            )

            candidates = analysis.get(
                "candidate_articles",
                []
            )

            keywords = analysis.get(
                "keywords",
                []
            )

            st.write(
                "**法的論点**"
            )

            st.write(
                issue
                if issue
                else "特定できませんでした。"
            )

            st.write(
                "**候補条文**"
            )

            if candidates:

                st.write(
                    " / ".join(
                        f"第{num}条"
                        for num in candidates
                    )
                )

            else:

                st.write(
                    "候補なし"
                )

            st.write(
                "**検索キーワード**"
            )

            if keywords:

                st.write(
                    " / ".join(
                        keywords
                    )
                )

            else:

                st.write(
                    "なし"
                )


        # -------------------------------------------------
        # 条文が見つからない
        # -------------------------------------------------

        if not selected:

            st.warning(
                "関連する地方自治法の条文を"
                "特定できませんでした。"
                "質問の表現を変えるか、"
                "条番号・制度名などを"
                "入力してください。"
            )

            st.stop()


        # -------------------------------------------------
        # 回答AIに渡す条文
        # -------------------------------------------------

        context = "\n\n".join(
            (
                f"第{num}条\n"
                f"{text[:4000]}"
            )
            for num, text
            in selected
        )


        # -------------------------------------------------
        # 最終回答
        # -------------------------------------------------

        with st.spinner(
            "条文を確認して"
            "AIが回答を作成しています..."
        ):

            response = (
                client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "あなたは地方自治法の"
                                "調査補助AIです。"

                                "利用者の質問に対し、"
                                "提示されたe-Gov由来の"
                                "地方自治法の条文を"
                                "主要な根拠として"
                                "回答してください。"

                                "提示された条文の中に"
                                "質問への直接の根拠が"
                                "存在するかを最初に"
                                "慎重に確認してください。"

                                "回答は原則として、"
                                "『結論』"
                                "『根拠となる条文』"
                                "『説明』"
                                "の順に整理してください。"

                                "根拠となる条番号と、"
                                "分かる場合は項番号も"
                                "明示してください。"

                                "候補として提示された条文が"
                                "質問と関係しない場合は、"
                                "無理にその条文を根拠に"
                                "回答しないでください。"

                                "提示された条文だけでは"
                                "判断できない場合は、"
                                "その旨を明確に"
                                "回答してください。"

                                "条文に書かれていない内容を"
                                "条文に書かれているかのように"
                                "断定してはいけません。"

                                "推測で条文内容を"
                                "補わないでください。"
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"質問：\n"
                                f"{question}\n\n"

                                f"検索AIが整理した法的論点：\n"
                                f"{analysis.get('issue', '')}\n\n"

                                f"参考条文：\n"
                                f"{context}"
                            )
                        }
                    ]
                )
            )


        # -------------------------------------------------
        # 回答表示
        # -------------------------------------------------

        st.subheader(
            "AIの回答"
        )

        answer = (
            response.choices[0]
            .message.content
            or "回答がありませんでした。"
        )

        st.write(
            answer
        )


        # -------------------------------------------------
        # 参照条文
        # -------------------------------------------------

        st.subheader(
            "参照した条文"
        )

        for num, text in selected:

            with st.expander(
                f"第{num}条"
            ):

                st.write(
                    text
                )


        # -------------------------------------------------
        # e-Gov
        # -------------------------------------------------

        st.markdown(
            f"[e-Govで地方自治法を確認]"
            f"({EGOV_URL})"
        )


        # -------------------------------------------------
        # 注意事項
        # -------------------------------------------------

        st.caption(
            "この回答はAIによる法令調査の"
            "参考情報です。"
            "AIによる条文候補の選択や回答には"
            "誤りが生じる可能性があります。"
            "重要な判断では、最新の法令、"
            "公的資料、判例・行政実例等を"
            "直接確認してください。"
        )


    except Exception as e:

        st.error(
            "処理中にエラーが発生しました。"
        )

        st.caption(
            f"{type(e).__name__}: {e}"
        )
