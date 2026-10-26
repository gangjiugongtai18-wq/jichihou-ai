
import json
import re
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from openai import OpenAI


# =========================================================
# 自治法AI Ver.4
# =========================================================

st.set_page_config(
    page_title="自治法AI Ver.4",
    page_icon="⚖️"
)

st.title("⚖️ 自治法AI")
st.caption("Ver.4｜地方自治法・自然言語検索")
st.info(
    "質問の意味をAIが分析し、"
    "e-Govの地方自治法本文を確認して回答します。"
)


# =========================================================
# 基本設定
# =========================================================

LAW_ID = "322AC0000000067"

LAW_URL = (
    f"https://laws.e-gov.go.jp/api/1/lawdata/{LAW_ID}"
)

EGOV_URL = (
    f"https://laws.e-gov.go.jp/law/{LAW_ID}"
)

MODEL_NAME = "gemini-3.5-flash-lite"

MAX_CANDIDATES = 12
MAX_SELECTED = 5
MAX_ARTICLE_CHARS = 6000


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

        num = article.get("Num", "").strip()

        text = "".join(
            article.itertext()
        ).strip()

        if num and text:
            articles.append((num, text))

    return articles


# =========================================================
# Geminiクライアント
# =========================================================

def get_ai_client():

    key = st.secrets.get("GEMINI_API_KEY")

    if not key:
        st.error(
            "Streamlit Secretsに"
            "GEMINI_API_KEYを設定してください。"
        )
        st.stop()

    return OpenAI(
        api_key=key,
        base_url=(
            "https://generativelanguage.googleapis.com/"
            "v1beta/openai/"
        ),
        timeout=60.0
    )


# =========================================================
# Gemini呼び出し
# =========================================================

def ask_gemini(client, system_prompt, user_prompt):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1
    )

    return (
        response.choices[0].message.content
        or ""
    )


# =========================================================
# JSON抽出
# =========================================================

def extract_json(text):

    if not text:
        return None

    text = text.strip()

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
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except (ValueError, TypeError):
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        return None

    try:
        result = json.loads(match.group(0))

        if isinstance(result, dict):
            return result

    except (ValueError, TypeError):
        pass

    return None


# =========================================================
# 条番号の正規化
# =========================================================

def normalize_article_number(value):

    if value is None:
        return None

    value = str(value).strip()

    # 第179条、第179条の2、179-2などに対応
    match = re.search(
        r"(\d+)(?:\s*(?:条の|の|-)\s*(\d+))?",
        value
    )

    if not match:
        return None

    main = str(int(match.group(1)))
    sub = match.group(2)

    if sub:
        return f"{main}_{int(sub)}"

    return main


def article_label(num):

    parts = str(num).split("_")

    if len(parts) == 2:
        return f"第{parts[0]}条の{parts[1]}"

    return f"第{num}条"


# =========================================================
# 質問に明示された条番号
# =========================================================

def get_direct_articles(question):

    matches = re.findall(
        r"第\s*(\d+)\s*条(?:の\s*(\d+))?",
        question
    )

    result = []

    for main, sub in matches:

        num = str(int(main))

        if sub:
            num += f"_{int(sub)}"

        if num not in result:
            result.append(num)

    if not result:

        if re.fullmatch(r"\s*\d+\s*", question):
            result.append(str(int(question.strip())))

    return result[:5]


# =========================================================
# AIによる質問分析
# =========================================================

def analyze_question(question, client):

    system_prompt = """
あなたは日本の地方自治法の条文検索を補助するAIです。

利用者の曖昧な日常語を法律概念に変換してください。

質問について、次を分析してください。

1. 中心となる法的論点
2. 合理的に考えられる複数の解釈
3. 各解釈に対応する法律概念
4. 関係する可能性がある地方自治法の条番号
5. 条文検索に使うキーワード

例：

「市長がいなくなったら誰が仕切るの？」

考えられる解釈：
・市長が欠けた場合の職務代理
・市長が一時的に職務を行えない場合の代理

法律概念：
・長が欠けたとき
・長に事故があるとき
・職務代理

候補条文：
・第152条

別の例：

「議会がごちゃごちゃして決めてくれない」

考えられる解釈：
・議会が議決しない場合
・議会が成立しない場合
・議会を招集する時間的余裕がない場合

候補条文：
・第179条

注意：
・候補条番号は推定にすぎません。
・条文の実在や内容は後続処理で確認します。
・確信できない条番号を無理に作らないでください。
・解釈は最大3件としてください。
・候補条文は最大8件としてください。

次のJSON形式だけを返してください。

{
  "issue": "中心となる法的論点",
  "interpretations": [
    {
      "meaning": "解釈",
      "legal_concepts": ["法律概念"],
      "candidate_articles": ["179"]
    }
  ],
  "candidate_articles": ["179", "180"],
  "keywords": ["専決処分", "議会", "議決"]
}

JSON以外の文章を書かないでください。
"""

    result = ask_gemini(
        client,
        system_prompt,
        question
    )

    data = extract_json(result)

    if not data:
        return {
            "issue": "",
            "interpretations": [],
            "candidate_articles": [],
            "keywords": []
        }

    issue = str(
        data.get("issue", "")
    ).strip()

    raw_interpretations = data.get(
        "interpretations", []
    )

    if not isinstance(raw_interpretations, list):
        raw_interpretations = []

    interpretations = []
    candidate_numbers = []
    keywords = []

    def add_number(value):

        num = normalize_article_number(value)

        if num and num not in candidate_numbers:
            candidate_numbers.append(num)

    def add_keyword(value):

        if not isinstance(value, str):
            return

        value = value.strip()

        if (
            2 <= len(value) <= 30
            and value not in keywords
        ):
            keywords.append(value)

    for item in raw_interpretations[:3]:

        if not isinstance(item, dict):
            continue

        meaning = str(
            item.get("meaning", "")
        ).strip()

        concepts = item.get(
            "legal_concepts", []
        )

        numbers = item.get(
            "candidate_articles", []
        )

        if not isinstance(concepts, list):
            concepts = []

        if not isinstance(numbers, list):
            numbers = []

        clean_concepts = []
        clean_numbers = []

        for concept in concepts[:5]:

            if isinstance(concept, str):
                concept = concept.strip()

                if concept:
                    clean_concepts.append(concept)
                    add_keyword(concept)

        for number in numbers[:5]:

            num = normalize_article_number(number)

            if num:
                clean_numbers.append(num)
                add_number(num)

        interpretations.append({
            "meaning": meaning,
            "legal_concepts": clean_concepts,
            "candidate_articles": clean_numbers
        })

    raw_numbers = data.get(
        "candidate_articles", []
    )

    if isinstance(raw_numbers, list):

        for number in raw_numbers[:8]:
            add_number(number)

    raw_keywords = data.get(
        "keywords", []
    )

    if isinstance(raw_keywords, list):

        for word in raw_keywords[:12]:
            add_keyword(word)

    return {
        "issue": issue,
        "interpretations": interpretations,
        "candidate_articles": candidate_numbers[:8],
        "keywords": keywords[:15]
    }


# =========================================================
# 条文番号で検索
# =========================================================

def get_articles_by_numbers(numbers, articles):

    wanted = set(numbers)

    return [
        (num, text)
        for num, text in articles
        if num in wanted
    ]


# =========================================================
# キーワードによる補助検索
# =========================================================

def keyword_search(keywords, articles, limit=8):

    scored = []

    for num, text in articles:

        score = 0
        matched = 0

        for word in keywords:

            if len(word) < 2:
                continue

            count = text.count(word)

            if count > 0:

                matched += 1

                score += min(count, 5)

                if len(word) >= 4:
                    score += 4

                elif len(word) >= 3:
                    score += 2

        if matched >= 2:
            score += matched * 3

        if score > 0:
            scored.append((score, num, text))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        (num, text)
        for _, num, text in scored[:limit]
    ]


# =========================================================
# 候補条文を統合
# =========================================================

def merge_articles(primary, secondary, limit=12):

    merged = []
    seen = set()

    for num, text in primary + secondary:

        if num in seen:
            continue

        seen.add(num)
        merged.append((num, text))

        if len(merged) >= limit:
            break

    return merged


# =========================================================
# AIによる関連性の再評価
# =========================================================

def rerank_articles(
    question,
    analysis,
    candidates,
    client
):

    if not candidates:
        return [], []

    context = "\n\n".join(
        (
            f"条番号：{article_label(num)}\n"
            f"条文本文：\n{text[:MAX_ARTICLE_CHARS]}"
        )
        for num, text in candidates
    )

    system_prompt = """
あなたは地方自治法の条文検索の審査担当AIです。

利用者の質問と、e-Govから取得された条文本文を比較し、
本当に回答の根拠になる条文を選別してください。

判定区分：

direct：
質問への直接の法的根拠となる条文

supporting：
直接の根拠を補足するために必要な条文

irrelevant：
質問への回答に不要な条文

uncertain：
関連性を確定できない条文

重要なルール：

・条番号の推定結果だけで採用しない。
・必ず提示された条文本文を確認する。
・文字列が一致するだけでは採用しない。
・質問に答えるために不要な条文は除外する。
・複数解釈がある場合は、それぞれを考慮する。
・判断できない条文は無理に採用しない。
・提示されていない条番号を追加しない。
・最大5条文を選択する。
・直接の根拠がなければ空配列を許容する。

次のJSON形式だけを返してください。

{
  "evaluations": [
    {
      "article": "179",
      "relevance": "direct",
      "reason": "議会が議決しない場合の規定"
    }
  ],
  "selected_articles": ["179"],
  "sufficient_evidence": true,
  "ambiguity_note": "複数解釈がある場合の説明"
}

JSON以外の文章を書かないでください。
"""

    user_prompt = (
        f"利用者の質問：\n{question}\n\n"
        f"推定された法的論点：\n"
        f"{analysis.get('issue', '')}\n\n"
        f"取得した条文：\n{context}"
    )

    result = ask_gemini(
        client,
        system_prompt,
        user_prompt
    )

    data = extract_json(result)

    if not data:
        return [], []

    valid_articles = {
        num: text
        for num, text in candidates
    }

    raw_evaluations = data.get(
        "evaluations", []
    )

    evaluations = []

    if isinstance(raw_evaluations, list):

        for item in raw_evaluations:

            if not isinstance(item, dict):
                continue

            num = normalize_article_number(
                item.get("article")
            )

            if num not in valid_articles:
                continue

            relevance = str(
                item.get("relevance", "")
            ).strip().lower()

            if relevance not in (
                "direct",
                "supporting",
                "irrelevant",
                "uncertain"
            ):
                relevance = "uncertain"

            evaluations.append({
                "article": num,
                "relevance": relevance,
                "reason": str(
                    item.get("reason", "")
                ).strip()
            })

    # AIのselected_articlesを無条件に信用せず、
    # evaluationsでdirect/supportingとされたものだけ採用
    approved = {
        item["article"]: item["relevance"]
        for item in evaluations
        if item["relevance"] in (
            "direct",
            "supporting"
        )
    }

    direct_numbers = [
        num
        for num, status in approved.items()
        if status == "direct"
    ]

    # 直接の根拠がなければ、
    # 補助条文だけで回答を作らない
    if not direct_numbers:
        return [], evaluations

    raw_selected = data.get(
        "selected_articles", []
    )

    if not isinstance(raw_selected, list):
        raw_selected = []

    selected_numbers = []

    for value in raw_selected:

        num = normalize_article_number(value)

        if (
            num in approved
            and num not in selected_numbers
        ):
            selected_numbers.append(num)

    # JSONの選択リストに直接根拠が
    # 欠落していても、評価結果を優先する
    for num in direct_numbers:

        if num not in selected_numbers:
            selected_numbers.insert(0, num)

    for num, status in approved.items():

        if (
            status == "supporting"
            and num not in selected_numbers
        ):
            selected_numbers.append(num)

    selected_numbers = selected_numbers[:MAX_SELECTED]

    selected = [
        (num, valid_articles[num])
        for num in selected_numbers
        if num in valid_articles
    ]

    return selected, evaluations


# =========================================================
# 最終回答の生成
# =========================================================

def generate_answer(
    question,
    analysis,
    selected,
    client
):

    context = "\n\n".join(
        (
            f"{article_label(num)}\n"
            f"{text[:MAX_ARTICLE_CHARS]}"
        )
        for num, text in selected
    )

    interpretations = analysis.get(
        "interpretations", []
    )

    interpretation_text = "\n".join(
        (
            f"・{item.get('meaning', '')}"
        )
        for item in interpretations
    )

    system_prompt = """
あなたは日本の地方自治法の調査補助AIです。

提示されたe-Gov由来の条文本文を根拠として、
日本語で分かりやすく回答してください。

厳守事項：

1. 提示されていない条文を根拠にしない。
2. 条文にない内容を断定しない。
3. 条番号と、確認できる項番号・号番号を示す。
4. 複数解釈が合理的に成立する場合は区別する。
5. 根拠が不足する場合は判断できないと説明する。
6. 質問に直接関係しない情報を増やさない。
7. 法令の引用とAIによる説明を区別する。

回答の基本構成：

【結論】
質問への簡潔な回答

【根拠条文】
根拠となる条番号・項番号・号番号

【説明】
条文の意味と適用上の注意

【解釈上の注意】
複数解釈や判断できない点があれば記載

条文本文にない具体的事実や
判例・行政実例を作り出してはいけません。
"""

    user_prompt = (
        f"質問：\n{question}\n\n"
        f"考えられる解釈：\n"
        f"{interpretation_text}\n\n"
        f"確認済みの根拠条文：\n"
        f"{context}"
    )

    return ask_gemini(
        client,
        system_prompt,
        user_prompt
    )


# =========================================================
# 画面：質問入力
# =========================================================

question = st.text_area(
    "地方自治法について質問してください",
    placeholder=(
        "例：議会がごちゃごちゃして"
        "何も決めてくれないとき、"
        "市長だけで決められないの？"
    ),
    height=120
)

search_button = st.button(
    "AIに質問する",
    type="primary"
)


# =========================================================
# メイン処理
# =========================================================

if search_button:

    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()

    try:

        client = get_ai_client()

        with st.spinner(
            "地方自治法を取得しています..."
        ):
            articles = get_law()

        if not articles:
            st.error(
                "地方自治法の条文を取得できませんでした。"
            )
            st.stop()

        # ---------------------------------------------
        # 1. 質問分析
        # ---------------------------------------------

        with st.spinner(
            "質問の意味を分析しています..."
        ):

            analysis = analyze_question(
                question,
                client
            )

        # 利用者が直接指定した条文は必ず候補に追加
        direct_numbers = get_direct_articles(
            question
        )

        candidate_numbers = list(
            direct_numbers
        )

        for num in analysis.get(
            "candidate_articles", []
        ):

            if num not in candidate_numbers:
                candidate_numbers.append(num)

        # ---------------------------------------------
        # 2. 候補条文の取得
        # ---------------------------------------------

        primary = get_articles_by_numbers(
            candidate_numbers,
            articles
        )

        secondary = keyword_search(
            analysis.get("keywords", []),
            articles,
            limit=8
        )

        candidates = merge_articles(
            primary,
            secondary,
            limit=MAX_CANDIDATES
        )

        # ---------------------------------------------
        # 3. 関連性の再評価
        # ---------------------------------------------

        with st.spinner(
            "候補条文の関連性を確認しています..."
        ):

            selected, evaluations = rerank_articles(
                question,
                analysis,
                candidates,
                client
            )

        # ---------------------------------------------
        # 4. 検索分析の表示
        # ---------------------------------------------

        with st.expander(
            "AIによる検索分析"
        ):

            st.write("**中心となる法的論点**")

            st.write(
                analysis.get("issue")
                or "特定できませんでした。"
            )

            st.write("**考えられる解釈**")

            interpretations = analysis.get(
                "interpretations", []
            )

            if interpretations:

                for item in interpretations:

                    st.write(
                        "・" + item.get(
                            "meaning", ""
                        )
                    )

            else:
                st.write("特定できませんでした。")

            st.write("**候補条文**")

            if candidates:

                st.write(
                    " / ".join(
                        article_label(num)
                        for num, _ in candidates
                    )
                )

            else:
                st.write("候補なし")

            st.write("**関連性の再評価**")

            if evaluations:

                for item in evaluations:

                    st.write(
                        f"{article_label(item['article'])}"
                        f"：{item['relevance']}"
                        f" — {item['reason']}"
                    )

            else:
                st.write("評価結果なし")

        # ---------------------------------------------
        # 5. 根拠不足の場合
        # ---------------------------------------------

        if not selected:

            st.warning(
                "取得した条文の中に、"
                "質問への直接の根拠を"
                "確認できませんでした。"
                "推測による法律上の結論は"
                "表示しません。"
            )

            st.markdown(
                f"[e-Govで地方自治法を確認]"
                f"({EGOV_URL})"
            )

            st.stop()

        # ---------------------------------------------
        # 6. 回答生成
        # ---------------------------------------------

        with st.spinner(
            "確認した条文を根拠に"
            "回答を作成しています..."
        ):

            answer = generate_answer(
                question,
                analysis,
                selected,
                client
            )

        # ---------------------------------------------
        # 7. 回答表示
        # ---------------------------------------------

        st.subheader("AIの回答")

        st.write(
            answer or "回答がありませんでした。"
        )

        # ---------------------------------------------
        # 8. 参照条文
        # ---------------------------------------------

        st.subheader("根拠として採用した条文")

        for num, text in selected:

            with st.expander(
                article_label(num)
            ):
                st.write(text)

        # ---------------------------------------------
        # 9. e-Govへのリンク
        # ---------------------------------------------

        st.markdown(
            f"[e-Govで地方自治法を確認]"
            f"({EGOV_URL})"
        )

        st.caption(
            "このサービスは法令調査の補助ツールです。"
            "AIによる条文検索・関連性判定・回答には"
            "誤りが生じる可能性があります。"
            "重要な判断では、最新の法令本文、"
            "判例、行政実例等を直接確認してください。"
        )

    except Exception as e:

        st.error(
            "処理中にエラーが発生しました。"
        )

        st.caption(
            f"エラーの種類：{type(e).__name__}"
        )
