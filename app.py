
import json
import re
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from openai import OpenAI

LAW_ID = "322AC0000000067"
LAW_URL = f"https://laws.e-gov.go.jp/law/{LAW_ID}"
API_URL = f"https://laws.e-gov.go.jp/api/1/lawdata/{LAW_ID}"
MODEL = "gemini-3.5-flash-lite"  # Ver.4と同じ設定。利用不可なら実際に使えるモデル名へ変更
MAX_CANDIDATES = 12
MAX_REFERENCES = 5

st.set_page_config(page_title="自治法AI Ver.5", page_icon="⚖️")
st.title("⚖️ 自治法AI Ver.5｜地方自治法・自然言語検索")
st.caption("質問の意味を分析し、e-Govの地方自治法本文を根拠に回答します。")


def normalize_num(value):
    """第179条の2 / 179-2 / 179_2 → 179_2"""
    s = str(value or "").strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    s = re.sub(r"[\s　]", "", s).replace("第", "").replace("条", "")
    s = s.replace("の", "_").replace("-", "_")
    return s if re.fullmatch(r"\d+(?:_\d+)*", s) else ""


def display_num(num):
    return "第" + num.replace("_", "条の", 1) + ("条" if "_" not in num else "")


def text_of(element):
    return "".join(element.itertext()).strip() if element is not None else ""


@st.cache_data(ttl=86400, show_spinner=False)
def load_law():
    response = requests.get(API_URL, timeout=35)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    articles = {}
    for node in root.findall(".//Article"):
        num = normalize_num(node.get("Num"))
        if not num or num in articles:
            continue
        title = text_of(node.find("ArticleTitle")) or display_num(num)
        paragraphs = []
        for para in node.findall("./Paragraph"):
            pnum = para.get("Num", "")
            sentence = text_of(para.find("ParagraphSentence"))
            if not sentence:
                sentence = text_of(para)
            items = []
            for item in para.findall("./Item"):
                label = text_of(item.find("ItemTitle"))
                body = text_of(item.find("ItemSentence")) or text_of(item)
                if body:
                    items.append({"label": label, "text": body})
            if sentence or items:
                paragraphs.append({"num": pnum, "text": sentence, "items": items})
        if not paragraphs:
            continue
        full = "\n".join(
            f"第{p['num']}項 {p['text']} " + " ".join(f"{x['label']} {x['text']}" for x in p['items'])
            for p in paragraphs
        )
        articles[num] = {"num": num, "title": title, "paragraphs": paragraphs, "text": full}
    if not articles:
        raise ValueError("e-Govの応答から条文を取得できませんでした。")
    return articles


def client_from_secrets():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("Streamlit SecretsにGEMINI_API_KEYがありません。")
    return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def call_ai(client, system, user, max_tokens=1800):
    result = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (result.choices[0].message.content or "").strip()


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("AIの分析結果がJSON形式ではありません。")
        return json.loads(raw[start:end + 1])


def article_numbers_in_question(question):
    q = question.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    return list(dict.fromkeys(
        n for n in (normalize_num(x) for x in re.findall(r"第\s*(\d+(?:条の\d+|の\d+)?)\s*条?", q)) if n
    ))


def analyze(question, client):
    system = """あなたは地方自治法の検索補助者です。法的結論は出さず、検索候補だけを作成します。
必ずJSONオブジェクトのみを返してください。形式:
{"issue":"質問の法律上の論点","interpretations":["解釈1","解釈2"],"candidate_articles":["179","180"],"keywords":["専決処分"]}
条番号は確信できるものだけ。『第179条の2』は『179_2』とする。
日常語から法的概念を考え、重要な関連条文を最大10件挙げてください。
定義を尋ねる質問では、その制度の根拠となる条文を優先してください。"""
    data = parse_json(call_ai(client, system, question, 1100))
    if not isinstance(data, dict):
        raise ValueError("検索分析の形式が不正です。")
    nums = [normalize_num(n) for n in data.get("candidate_articles", []) if isinstance(n, (str, int))]
    data["candidate_articles"] = list(dict.fromkeys(n for n in nums if n))[:10]
    data["keywords"] = [str(k).strip() for k in data.get("keywords", []) if isinstance(k, str) and len(k.strip()) >= 2][:8]
    data["issue"] = str(data.get("issue", ""))
    data["interpretations"] = [str(x) for x in data.get("interpretations", []) if isinstance(x, str)][:3]
    return data


def keyword_candidates(articles, keywords, limit=6):
    scored = []
    for num, article in articles.items():
        body = article["text"]
        matches = [k for k in keywords if k in body]
        if matches:
            score = len(matches) * 4 + sum(min(body.count(k), 3) for k in matches)
            scored.append((score, num))
    scored.sort(reverse=True)
    return [num for _, num in scored[:limit]]


def choose_candidates(question, analysis, articles):
    explicit = article_numbers_in_question(question)
    ai_nums = analysis["candidate_articles"]
    # 重要な条文をキーワード検索の結果で押し出さない
    ordered = explicit + ai_nums + keyword_candidates(articles, analysis["keywords"])
    return list(dict.fromkeys(n for n in ordered if n in articles))[:MAX_CANDIDATES]


def evidence_text(article, limit=4500):
    return article["title"] + "\n" + article["text"][:limit]


def rerank(question, analysis, candidate_nums, articles, client):
    material = "\n\n".join(f"[{n}]\n{evidence_text(articles[n])}" for n in candidate_nums)
    system = """あなたは地方自治法の条文選別者です。与えられたe-Gov条文だけで判断します。
JSONのみを返すこと。形式:
{"direct":["179"],"supporting":["180"],"reason":"選別理由","needs_clarification":false}
質問への直接の根拠となる条文をdirect、理解に役立つ別の条文をsupportingとする。
制度の定義を尋ねられたときは、その制度を規定する条文をdirectに含める。
根拠が明確な条文を慎重すぎる理由で除外しないこと。
無関係な条文は採用しないこと。提示されていない条番号は返さないこと。
本文の一部が省略されている場合は、その範囲外の内容を推測しないこと。"""
    result = parse_json(call_ai(client, system, f"質問: {question}\n論点: {analysis['issue']}\n\n候補条文:\n{material}", 1200))
    allowed = set(candidate_nums)
    direct = [normalize_num(x) for x in result.get("direct", []) if isinstance(x, (str, int))]
    supporting = [normalize_num(x) for x in result.get("supporting", []) if isinstance(x, (str, int))]
    direct = list(dict.fromkeys(n for n in direct if n in allowed))
    supporting = list(dict.fromkeys(n for n in supporting if n in allowed and n not in direct))
    return direct[:MAX_REFERENCES], supporting[:MAX_REFERENCES], str(result.get("reason", ""))


def generate_answer(question, analysis, direct, supporting, articles, client):
    refs = (direct + supporting)[:MAX_REFERENCES]
    evidence = "\n\n".join(f"[{n}]\n{evidence_text(articles[n], 6500)}" for n in refs)
    system = """あなたは地方自治法の説明担当です。提示されたe-Gov条文以外を法律上の根拠にしてはいけません。
JSONオブジェクトのみを返してください。形式:
{"conclusion":"簡潔な結論","explanation":"やさしく正確な説明","conditions":["適用条件"],"cautions":["例外や注意点"],"citations":[{"article":"179","paragraph":"1","item":"","claim":"この条項が支える結論"}],"unsupported":"条文だけでは判断できないこと。なければ空文字"}
守ること:
1. 『できる』『しなければならない』『求める』『得る』『代理』『選任』の法律上の違いを維持。
2. 条文の引用文を生成しない。原文はアプリが直接表示する。
3. 原則・条件・例外・事後手続きを混同しない。
4. 質問に直接答え、冗長な脱線を避ける。
5. 条文にない事実、判例、運用、断定は追加しない。
6. 根拠のないことはunsupportedに記載。
7. citationsのarticleは提供された条番号のみ。項・号が確認できない場合は空文字。"""
    raw = call_ai(client, system, f"質問: {question}\n論点: {analysis['issue']}\n直接の根拠: {direct}\n参考: {supporting}\n\n条文:\n{evidence}", 2300)
    result = parse_json(raw)
    if not isinstance(result, dict) or not result.get("conclusion"):
        raise ValueError("回答を構造化できませんでした。")
    result["citations"] = [c for c in result.get("citations", []) if isinstance(c, dict) and normalize_num(c.get("article")) in refs]
    return result


def verify_answer(question, answer, direct, supporting, articles, client):
    refs = (direct + supporting)[:MAX_REFERENCES]
    evidence = "\n\n".join(f"[{n}]\n{evidence_text(articles[n], 6500)}" for n in refs)
    system = """地方自治法の回答を条文と照合する独立チェック担当です。JSONのみ:
{"ok":true,"problems":[],"revised_conclusion":"","revised_explanation":""}
義務と権限、承認を求める/得る、要件、例外、職務代理と就任、条文にない断定を厳しく確認。
問題があればok=false、problemsに具体的に記載。修正可能ならrevised_*に条文に忠実な文を書く。
与えられた条文にない事実を補わない。"""
    return parse_json(call_ai(client, system, "質問:\n" + question + "\n回答:\n" + json.dumps(answer, ensure_ascii=False) + "\n条文:\n" + evidence, 1300))


def show_article(article):
    st.markdown(f"**{article['title']}**")
    for p in article["paragraphs"]:
        prefix = f"第{p['num']}項" if p["num"] else "本文"
        st.markdown(f"**{prefix}**　{p['text']}")
        for item in p["items"]:
            st.write(f"{item['label']}　{item['text']}")


question = st.text_area("地方自治法について質問してください", placeholder="例：専決処分とは何ですか？", height=90)
if st.button("根拠条文を調べる", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("e-Govの条文を取得しています..."):
            articles = load_law()
            client = client_from_secrets()
        with st.spinner("質問の意味と候補条文を分析しています..."):
            analysis = analyze(question, client)
            candidate_nums = choose_candidates(question, analysis, articles)
        with st.expander("AIによる検索分析"):
            st.write("**論点：**", analysis["issue"])
            st.write("**考えられる意味：**", "／".join(analysis["interpretations"]) or "特になし")
            st.write("**候補条文：**", "、".join(display_num(n) for n in candidate_nums) or "なし")
        if not candidate_nums:
            st.warning("質問に対応する条文候補を取得できませんでした。表現を変えて再検索してください。")
            st.stop()
        with st.spinner("取得した条文から直接の根拠を選別しています..."):
            direct, supporting, reason = rerank(question, analysis, candidate_nums, articles, client)
        if not direct:
            st.warning("取得した条文の中に、質問への直接の根拠を確認できませんでした。推測による法律上の結論は表示しません。")
            with st.expander("確認した候補条文"):
                st.write("、".join(display_num(n) for n in candidate_nums))
                st.write(reason)
            st.stop()
        with st.spinner("根拠に基づく回答を作成しています..."):
            answer = generate_answer(question, analysis, direct, supporting, articles, client)
        with st.spinner("回答の法律表現を再確認しています..."):
            check = verify_answer(question, answer, direct, supporting, articles, client)
        st.subheader("AIの回答")
        if check.get("ok") is False:
            st.warning("回答の整合性チェックで注意点が見つかりました。未検証の結論は表示しません。")
            for problem in check.get("problems", []):
                st.write("・", str(problem))
        else:
            st.markdown("### 結論")
            st.write(str(answer["conclusion"]))
            st.markdown("### 根拠条文")
            for c in answer["citations"]:
                n = normalize_num(c.get("article"))
                p = str(c.get("paragraph", "")).strip()
                i = str(c.get("item", "")).strip()
                label = display_num(n) + (f"第{p}項" if p.isdigit() else "") + (f"第{i}号" if i.isdigit() else "")
                st.write(f"・{label}：{c.get('claim', '')}")
            st.markdown("### 説明")
            st.write(str(answer.get("explanation", "")))
            if answer.get("conditions"):
                st.markdown("### 適用条件")
                for s in answer["conditions"]:
                    st.write("・", str(s))
            if answer.get("cautions"):
                st.markdown("### 注意点")
                for s in answer["cautions"]:
                    st.write("・", str(s))
            if answer.get("unsupported"):
                st.info("条文だけでは判断できないこと：" + str(answer["unsupported"]))
        st.subheader("e-Govから取得した根拠条文の原文")
        st.caption("以下はAIが書き直した引用ではなく、取得したXMLの条文テキストです。条・項・号の構造を表示します。")
        for n in direct:
            with st.expander(f"【直接の根拠】{display_num(n)}", expanded=True):
                show_article(articles[n])
        for n in supporting[:max(0, MAX_REFERENCES - len(direct))]:
            with st.expander(f"【参考条文】{display_num(n)}"):
                show_article(articles[n])
        st.link_button("e-Govで地方自治法を確認", LAW_URL)
        st.caption("本サービスは法令調査の補助です。個別事案の適法性を保証するものではありません。")
    except Exception as exc:
        st.error(f"処理できませんでした：{type(exc).__name__}：{exc}")
        st.info("Geminiの利用制限、モデル名、e-Govの応答、AIのJSON形式などを確認してください。")
