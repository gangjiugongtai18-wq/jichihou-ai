import json
import re
import datetime as dt
import xml.etree.ElementTree as ET
from collections import defaultdict

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="自治法AI Ver.6", page_icon="⚖️", layout="wide")
st.title("⚖️ 自治法AI Ver.6｜地方自治法・施行令・施行規則")
st.caption("e-Govの施行中本文を根拠に検索。未施行改正と委任関係は確認できた範囲だけ表示します。")

LAWS = {
    "法": ("地方自治法", "322AC0000000067"),
    "令": ("地方自治法施行令", "322CO0000000016"),
    "規則": ("地方自治法施行規則", "322M40000008029"),
}
API1 = "https://laws.e-gov.go.jp/api/1/lawdata/"
API2 = "https://laws.e-gov.go.jp/api/2/law_revisions/"
MODEL = "gemini-3.5-flash-lite"  # 利用中のモデル名。接続できない場合は有効なモデルに変更
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
KANJI_NUMS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
              "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20}


def number(value):
    s = str(value or "").translate(DIGITS).strip().replace("条", "").replace("第", "")
    s = s.replace("の", "_").replace("-", "_")
    return s if re.fullmatch(r"\d+(?:_\d+)*", s) else ""


def label(num):
    return "第" + num.replace("_", "条の", 1) + ("" if "_" in num else "条")


def content(node):
    return "".join(node.itertext()).strip() if node is not None else ""


def item_label_to_number(s):
    s = str(s).replace("第", "").replace("号", "").strip().translate(DIGITS)
    if s.isdigit():
        return int(s)
    return KANJI_NUMS.get(s)


@st.cache_data(ttl=43200, show_spinner=False)
def get_law(kind):
    name, law_id = LAWS[kind]
    response = requests.get(API1 + law_id, timeout=50)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    articles = {}
    # 附則の同番号条文と混同しないよう本則の条文だけを優先
    main = root.find(".//MainProvision")
    if main is None:
        raise ValueError(f"{name}: 本則のXMLを解析できません")
    for article in main.findall(".//Article"):
        num = number(article.get("Num"))
        if not num or num in articles:
            continue
        paragraphs = []
        for p in article.findall("./Paragraph"):
            pnum = str(p.get("Num") or len(paragraphs) + 1)
            sentence = content(p.find("ParagraphSentence"))
            items = []
            for item in p.findall("./Item"):
                item_num = item_label_to_number(item.get("Num") or content(item.find("ItemTitle")))
                item_text = content(item.find("ItemSentence"))
                if item_text:
                    items.append({"num": item_num, "text": item_text})
            if sentence or items:
                paragraphs.append({"num": pnum, "text": sentence, "items": items})
        if paragraphs:
            plain = "\n".join(
                f"第{p['num']}項 {p['text']} " + " ".join(f"第{x['num']}号 {x['text']}" for x in p['items'])
                for p in paragraphs
            )
            articles[num] = {"num": num, "paragraphs": paragraphs, "text": plain}
    if not articles:
        raise ValueError(f"{name}: 条文が取得できませんでした")
    return articles


@st.cache_data(ttl=43200, show_spinner=False)
def revision_status(kind):
    """法令単位の未施行履歴。条項単位の施行判定には流用しない。"""
    law_id = LAWS[kind][1]
    try:
        res = requests.get(API2 + law_id, timeout=18, headers={"Accept": "application/json"})
        res.raise_for_status()
        data = res.json()
        today = dt.date.today()
        found = []

        def walk(obj):
            if isinstance(obj, dict):
                # APIの履歴情報に明示された施行日だけを抽出する
                for key in ("enforcement_date", "enforcementDate"):
                    val = obj.get(key)
                    if isinstance(val, str):
                        m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", val)
                        if m:
                            try:
                                day = dt.date(*map(int, m.groups()))
                                if day > today:
                                    found.append(day.isoformat())
                            except ValueError:
                                pass
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for v in obj:
                    walk(v)
        walk(data)
        return sorted(set(found)), "取得済み"
    except Exception:
        return [], "取得不可"


def ai_client():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Streamlit SecretsにGEMINI_API_KEYを登録してください")
    return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def ask(client, system, user, limit=1600):
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=limit,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def as_json(raw):
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b < a:
            raise ValueError("AIの応答をJSONとして解析できません")
        return json.loads(raw[a:b + 1])


def explicit_refs(question):
    q = question.translate(DIGITS)
    result = []
    # 第3項だけを「第3条」と誤認しないよう、必ず「条」を要求
    pat = r"第\s*(\d+(?:の\d+)?)\s*条(?:\s*第\s*(\d+)\s*項)?(?:\s*第\s*(\d+)\s*号)?"
    for m in re.finditer(pat, q):
        before = q[max(0, m.start() - 25):m.start()]
        kind = "規則" if "施行規則" in before else "令" if "施行令" in before else "法"
        result.append({"kind": kind, "article": number(m.group(1)),
                       "paragraph": m.group(2), "item": m.group(3)})
    return result


def search_articles(question, corpus, limit=12):
    # 検索分析は候補作成専用。法的結論はAIに検索時点で書かせない
    keywords = re.findall(r"[一-龥ぁ-んァ-ヶ]{2,8}", question)
    keywords += [x for x in ("専決", "議会", "副市長", "条例", "予算", "出張", "承認", "代理") if x in question]
    scored = []
    for kind, articles in corpus.items():
        for num, art in articles.items():
            score = sum(min(3, art["text"].count(w)) for w in set(keywords) if len(w) >= 2)
            if score:
                scored.append((score, kind, num))
    scored.sort(reverse=True)
    return [{"kind": k, "article": n} for _, k, n in scored[:limit]]


def candidate_refs(question, corpus, client):
    explicit = explicit_refs(question)
    system = """あなたは法令検索の候補抽出器です。法的結論を出してはいけません。
地方自治法・地方自治法施行令・地方自治法施行規則の候補条番号をJSONのみで返す。
形式: {"candidates":[{"kind":"法","article":"179"}],"keywords":["専決処分"]}
kindは法/令/規則のみ。条番号は数字、枝番号は179_2の形式。第3項を第3条と解釈しない。
確信がない候補は出さない。"""
    try:
        result = as_json(ask(client, system, question, 900))
        ai_refs = result.get("candidates", [])
    except Exception:
        ai_refs = []
    merged, seen = [], set()
    for x in explicit + ai_refs + search_articles(question, corpus):
        if not isinstance(x, dict):
            continue
        k, n = x.get("kind"), number(x.get("article"))
        if k not in corpus or n not in corpus[k] or (k, n) in seen:
            continue
        seen.add((k, n))
        merged.append({"kind": k, "article": n})
        if len(merged) >= 14:
            break
    return merged, explicit


def paragraph_view(kind, num, art, only_paragraph=None, only_item=None):
    lines = [f"【{LAWS[kind][0]} {label(num)}】"]
    for p in art["paragraphs"]:
        if only_paragraph and p["num"] != str(only_paragraph):
            continue
        lines.append(f"第{p['num']}項 {p['text']}")
        for item in p["items"]:
            if only_item and item["num"] != int(only_item):
                continue
            lines.append(f"  第{item['num']}号 {item['text']}")
    return "\n".join(lines)


def related_provisions(selected, corpus):
    """下位法令に親法令の条番号が明記される場合だけ候補として提示。委任確定はしない。"""
    results = []
    for ref in selected:
        kind, num = ref["kind"], ref["article"]
        if kind == "規則":
            continue
        target = "令" if kind == "法" else "規則"
        # 法律条文の引用「第百七十九条」等の漢数字もあるため、AIに確定判定させず候補のみ
        numeral = num.split("_")[0]
        expressions = [f"第{numeral}条", f"第{numeral}條"]
        for target_num, art in corpus.get(target, {}).items():
            if any(x in art["text"] for x in expressions):
                results.append({"kind": target, "article": target_num,
                                "source": f"{LAWS[kind][0]} {label(num)} の番号への言及候補"})
            if len(results) >= 6:
                return results
    return results


def critical_errors(question, answer, refs):
    text = " ".join(str(answer.get(k, "")) for k in ("conclusion", "explanation", "conditions", "cautions"))
    problems = []
    if any(r["kind"] == "法" and r["article"] == "179" for r in refs):
        if re.search(r"承認を(?:必ず)?得なければならな|承認を得る必要があ|承認を受けなければならな", text):
            problems.append("第179条第3項は『承認を求めなければならない』と規定しています。『得なければならない』への置換を検出しました。")
    if "出張" in question and re.search(r"必ず.{0,12}(代理|副市長)|出張.{0,12}当然.{0,12}代理", text):
        problems.append("出張だけで当然に職務代理になると断定しています。")
    return problems


def answer_question(question, selected, corpus, client, explicit):
    passages = []
    for ref in selected[:7]:
        kind, num = ref["kind"], ref["article"]
        matching = next((r for r in explicit if r["kind"] == kind and r["article"] == num), None)
        # 指定条項があればその項を優先。ただし他項も必要な質問ではAIに不足を明示させる
        passages.append(paragraph_view(kind, num, corpus[kind][num],
                                       matching.get("paragraph") if matching else None,
                                       matching.get("item") if matching else None)[:9000])
    evidence = "\n\n".join(passages)[:26000]
    system = """あなたは日本の地方自治法に関する厳格な法令説明者です。提供したe-Gov本文だけを根拠に回答してください。
・義務「求めなければならない」と結果「得なければならない」を絶対に混同しない。
・「できる」「しなければならない」「してはならない」を意味が変わる形で言い換えない。
・質問に「必ず」があれば、条件を確認して最初に「はい」「いいえ」「一概にはいえません」で答える。
・未施行の法令内容、判例、通達、委任関係は本文から確定できなければ断定しない。
・引用する根拠は提供した条文の条・項・号のみ。確認できない事項は確認できないと書く。
・法令に規定のない事実や手続きを補わない。
JSONのみを返す。形式:
{"conclusion":"...","explanation":"...","conditions":["..."],"cautions":["..."],
"citations":[{"kind":"法","article":"179","paragraph":"3","item":null}]}
引用のkindは法/令/規則。根拠として直接使用したものだけを列挙。"""
    raw = ask(client, system, f"質問：{question}\n\n確認済み条文：\n{evidence}", 2300)
    answer = as_json(raw)
    if not isinstance(answer, dict):
        raise ValueError("回答形式が不正です")
    citations = []
    for x in answer.get("citations", []):
        if not isinstance(x, dict):
            continue
        k, n = x.get("kind"), number(x.get("article"))
        if k not in corpus or n not in corpus[k] or not any(r["kind"] == k and r["article"] == n for r in selected[:7]):
            continue
        p = str(x.get("paragraph") or "")
        if p and p not in {a["num"] for a in corpus[k][n]["paragraphs"]}:
            continue
        citations.append({"kind": k, "article": n, "paragraph": p, "item": x.get("item")})
    answer["citations"] = citations
    errors = critical_errors(question, answer, selected[:7])
    if not citations:
        errors.append("有効な根拠条文の引用を確認できませんでした。")
    return answer, errors


with st.sidebar:
    st.header("検索設定")
    kinds = st.multiselect("検索する法令", list(LAWS), default=list(LAWS),
                           format_func=lambda k: LAWS[k][0])
    st.caption("本文はe-Gov API Version 1の施行中法令を取得します。")
    st.caption("未施行の改正履歴はVersion 2で法令単位に確認します。条項単位の判定ではありません。")
    st.markdown("[e-Gov 法令API Version 2](https://laws.e-gov.go.jp/api/2/swagger-ui/)")

question = st.text_area("地方自治法について質問してください", placeholder="例：地方自治法第179条第3項には何が書かれていますか？", height=95)
run = st.button("条文を検索して回答", type="primary", disabled=not kinds)

if run:
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()
    try:
        with st.spinner("e-Govの3法令を取得しています..."):
            corpus = {k: get_law(k) for k in kinds}
        with st.spinner("条文を検索・照合しています..."):
            client = ai_client()
            selected, explicit = candidate_refs(question, corpus, client)
        if not selected:
            st.warning("根拠となる条文を特定できませんでした。質問を具体的にしてください。")
            st.stop()

        st.subheader("条文検索結果")
        st.write("、".join(f"{LAWS[r['kind']][0]} {label(r['article'])}" for r in selected[:7]))
        with st.spinner("根拠条文から回答を作成しています..."):
            answer, errors = answer_question(question, selected, corpus, client, explicit)

        st.subheader("AIの回答")
        if errors:
            st.error("法律表現または根拠の検証で問題を検出しました。AIの解釈文は表示せず、下の条文原文を確認してください。")
            for e in errors:
                st.write("・" + e)
        else:
            st.markdown("**結論**")
            st.write(answer.get("conclusion", ""))
            st.markdown("**説明**")
            st.write(answer.get("explanation", ""))
            for heading, key in (("適用条件", "conditions"), ("注意点", "cautions")):
                vals = answer.get(key, [])
                if isinstance(vals, list) and vals:
                    st.markdown(f"**{heading}**")
                    for v in vals:
                        st.write("・" + str(v))
            st.markdown("**回答に用いた根拠条文**")
            for c in answer["citations"]:
                suffix = f"第{c['paragraph']}項" if c["paragraph"] else ""
                if c["item"]:
                    suffix += f"第{c['item']}号"
                st.write(f"・{LAWS[c['kind']][0]} {label(c['article'])}{suffix}")

        st.subheader("e-Govから取得した条文原文")
        st.caption("AIによる書換えではありません。第1項・第2項などの構造を保持しています。")
        for i, r in enumerate(selected[:7]):
            kind, num = r["kind"], r["article"]
            name, law_id = LAWS[kind]
            with st.expander(f"{name} {label(num)}", expanded=(i == 0)):
                match = next((x for x in explicit if x["kind"] == kind and x["article"] == num), None)
                st.text(paragraph_view(kind, num, corpus[kind][num],
                                       match.get("paragraph") if match else None,
                                       match.get("item") if match else None))
                st.link_button("e-Govで法令を確認", f"https://laws.e-gov.go.jp/law/{law_id}")

        st.subheader("施行状況")
        st.info("上の本文はe-Gov API Version 1から取得した施行中本文です。未施行改正後の条文は混在させていません。")
        for kind in kinds:
            dates, state = revision_status(kind)
            if dates:
                st.warning(f"{LAWS[kind][0]}：法令履歴に将来の施行日があります（{', '.join(dates[:5])}）。該当条項は未判定です。")
            elif state == "取得不可":
                st.caption(f"{LAWS[kind][0]}：未施行改正の履歴を取得できません。未施行がないという意味ではありません。")
            else:
                st.caption(f"{LAWS[kind][0]}：取得した履歴から将来の施行日を検出しませんでした。個別条項の施行状況は未判定です。")

        st.subheader("施行令・施行規則との関連")
        related = related_provisions(selected[:5], corpus)
        if related:
            st.caption("以下は条番号への言及に基づく検索候補です。委任関係の確定ではありません。")
            for x in related:
                st.write(f"・候補：{LAWS[x['kind']][0]} {label(x['article'])}")
                with st.expander("候補条文の原文を確認"):
                    st.text(paragraph_view(x["kind"], x["article"], corpus[x["kind"]][x["article"]]))
        else:
            st.caption("対応する施行令・施行規則の条文を自動では特定できませんでした。関連規定が存在しないという意味ではありません。")
    except Exception as exc:
        st.error("取得または回答処理に失敗しました。")
        st.code(str(exc))
