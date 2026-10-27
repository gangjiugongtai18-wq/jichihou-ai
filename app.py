
import datetime as dt
import json
import re
import xml.etree.ElementTree as ET

import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="⚖️ 自治法AI Ver.7.3", page_icon="⚖️", layout="wide")
st.title("⚖️ 自治法AI Ver.7.3")
st.caption("e-Govの法令原文を確認し、自治体の条例・規則等で別途確認すべき事項を表示します。")

LAWS = {
    "法": ("地方自治法", "322AC0000000067"),
    "令": ("地方自治法施行令", "322CO0000000016"),
    "規則": ("地方自治法施行規則", "322M40000008029"),
}
API1 = "https://laws.e-gov.go.jp/api/1/lawdata/"
API2 = "https://laws.e-gov.go.jp/api/2/law_revisions/"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
LOCAL_PATTERNS = [
    (r"(?:普通地方公共団体|地方公共団体|当該団体|当該普通地方公共団体)?の?条例(?:で|により|の定めるところ)", "条例"),
    (r"(?:普通地方公共団体|地方公共団体|当該団体|当該普通地方公共団体)の規則(?:で|により|の定めるところ)", "自治体の規則"),
    (r"(?:議会の)?会議規則(?:で|により|の定めるところ)", "議会の会議規則"),
    (r"(?:普通地方公共団体の)?長が(?:別に)?定める", "長が定める規程等"),
    (r"(?:条例|規則)の定めるところ", "条例・規則"),
]
# 「規則で定める」だけでは国の省令を指す場合もあるため、自治体の規則と即断しない。
AMBIGUOUS_PATTERN = r"(?<!省令)(?<!総務省令)規則で定める"
NATIONAL_REF_PATTERN = re.compile(r"([一-龥ぁ-んァ-ヶ・]{2,55}(?:法律|法|政令|省令|施行令|施行規則))第[一二三四五六七八九十百千〇零０-９0-9の]+条")


def normalize_num(value):
    s = str(value or "").translate(DIGITS).strip().replace("第", "").replace("条", "")
    s = s.replace("の", "_").replace("-", "_")
    return s if re.fullmatch(r"\d+(?:_\d+)*", s) else ""


def article_label(num):
    parts = num.split("_")
    return "第" + parts[0] + "条" + "".join("の" + x for x in parts[1:])


def node_text(node):
    return "".join(node.itertext()).strip() if node is not None else ""


def parse_article(article):
    paragraphs = []
    for p in article.findall("./Paragraph"):
        pnum = str(p.get("Num") or len(paragraphs) + 1)
        sentence = node_text(p.find("./ParagraphSentence"))
        items = []
        for item in p.findall("./Item"):
            item_num = str(item.get("Num") or "").translate(DIGITS)
            item_num = item_num.replace("第", "").replace("号", "")
            # 号の内部の細目も、少なくとも本文として保存する。
            item_text = node_text(item.find("./ItemSentence"))
            subitems = [node_text(x) for x in item if x.tag not in ("ItemTitle", "ItemSentence")]
            item_text += " " + " ".join(x for x in subitems if x)
            if item_text.strip():
                items.append({"num": item_num, "text": item_text.strip()})
        if sentence or items:
            paragraphs.append({"num": pnum, "text": sentence, "items": items})
    return paragraphs


def render_article(name, num, art, paragraph=None, item=None):
    lines = [f"【{name} {article_label(num)}】"]
    for p in art["paragraphs"]:
        if paragraph and p["num"] != str(paragraph):
            continue
        lines.append(f"第{p['num']}項 {p['text']}")
        for it in p["items"]:
            if item and it["num"] != str(item):
                continue
            lines.append(f"  第{it['num']}号 {it['text']}")
    return "\n".join(lines)


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_law(law_id):
    res = requests.get(API1 + law_id, timeout=55)
    res.raise_for_status()
    root = ET.fromstring(res.content)
    main = root.find(".//MainProvision")
    if main is None:
        raise ValueError(f"法令ID {law_id} の本則を取得できません")
    articles = {}
    for node in main.findall(".//Article"):
        num = normalize_num(node.get("Num"))
        if not num or num in articles:
            continue
        paragraphs = parse_article(node)
        if paragraphs:
            plain = "\n".join(
                p["text"] + " " + " ".join(i["text"] for i in p["items"])
                for p in paragraphs
            )
            articles[num] = {"paragraphs": paragraphs, "text": plain}
    if not articles:
        raise ValueError(f"法令ID {law_id} の条文が見つかりません")
    return articles


@st.cache_data(ttl=43200, show_spinner=False)
def revision_status(law_id):
    try:
        r = requests.get(API2 + law_id, timeout=20, headers={"Accept": "application/json"})
        r.raise_for_status()
        data = r.json()
        future = []

        def walk(obj):
            if isinstance(obj, dict):
                for key in ("enforcement_date", "enforcementDate"):
                    v = obj.get(key)
                    if isinstance(v, str):
                        m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", v)
                        if m:
                            try:
                                d = dt.date(*map(int, m.groups()))
                                if d > dt.date.today():
                                    future.append(d.isoformat())
                            except ValueError:
                                pass
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, list):
                for value in obj:
                    walk(value)

        walk(data)
        return sorted(set(future)), True
    except Exception:
        return [], False


def client_for_gemini():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Streamlit SecretsにGEMINI_API_KEYが設定されていません")
    return OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")


def ai_json(client, model, system, message, max_tokens=1800):
    r = client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": message}],
    )
    raw = (r.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        a, b = raw.find("{"), raw.rfind("}")
        if a < 0 or b < a:
            raise ValueError("AIのJSON応答を解析できません")
        return json.loads(raw[a:b + 1])


def explicit_refs(question, law_names):
    q = question.translate(DIGITS)
    pat = r"第\s*(\d+(?:の\d+)?)\s*条(?:\s*第\s*(\d+)\s*項)?(?:\s*第\s*(\d+)\s*号)?"
    result = []
    for m in re.finditer(pat, q):
        before = q[max(0, m.start() - 55):m.start()]
        hits = [(len(name), k) for k, (name, _) in law_names.items() if name in before]
        kind = max(hits)[1] if hits else "法"
        result.append({"kind": kind, "article": normalize_num(m.group(1)),
                       "paragraph": m.group(2) or "", "item": m.group(3) or ""})
    return result


def find_candidates(question, corpus, names, client, model):
    explicit = explicit_refs(question, names)
    system = """あなたは法令検索用の条番号候補抽出器です。法的結論は出さないでください。
JSONのみ: {"candidates":[{"kind":"法","article":"234"}]}
kindは入力で示す識別子のみ。条番号は半角数字、枝番号は167_2形式。
確信のない条番号を捏造しない。"""
    try:
        ai = ai_json(client, model, system, "法令識別子: " + ", ".join(f"{k}={v[0]}" for k, v in names.items()) + "\n質問: " + question, 900)
        ai_refs = ai.get("candidates", [])
    except Exception:
        ai_refs = []
    words = set(re.findall(r"[一-龥ぁ-んァ-ヶ]{2,7}", question))
    words |= {w for w in ("専決", "承認", "随意契約", "入札", "条例", "議会", "副市長", "代理", "契約") if w in question}
    ranked = []
    for kind, articles in corpus.items():
        for num, art in articles.items():
            score = sum(min(3, art["text"].count(w)) for w in words if len(w) > 1)
            if score:
                ranked.append((score, kind, num))
    ranked.sort(key=lambda x: -x[0])
    refs, seen = [], set()
    for x in explicit + ai_refs + [{"kind": k, "article": n} for _, k, n in ranked[:12]]:
        if not isinstance(x, dict):
            continue
        k, n = x.get("kind"), normalize_num(x.get("article"))
        if k not in corpus or n not in corpus[k] or (k, n) in seen:
            continue
        seen.add((k, n))
        refs.append({"kind": k, "article": n})
        if len(refs) >= 10:
            break
    return refs, explicit


def local_checks(refs, corpus, names):
    checks, seen = [], set()
    for ref in refs:
        k, n = ref["kind"], ref["article"]
        art = corpus[k][n]
        for p in art["paragraphs"]:
            chunks = [(p["text"], f"第{p['num']}項")]
            chunks += [(it["text"], f"第{p['num']}項第{it['num']}号") for it in p["items"]]
            for body, loc in chunks:
                kinds = [name for pattern, name in LOCAL_PATTERNS if re.search(pattern, body)]
                if not kinds and re.search(AMBIGUOUS_PATTERN, body):
                    kinds = ["規則（制定主体の確認が必要）"]
                for name in set(kinds):
                    key = (k, n, loc, name)
                    if key not in seen:
                        seen.add(key)
                        checks.append({"source": f"{names[k][0]} {article_label(n)}{loc}",
                                       "type": name, "excerpt": body[:230]})
    return checks


def national_mentions(refs, corpus, names):
    results = set()
    for ref in refs:
        body = corpus[ref["kind"]][ref["article"]]["text"]
        for m in NATIONAL_REF_PATTERN.finditer(body):
            candidate = m.group(0)
            if len(candidate) < 85:
                results.add(candidate)
    return sorted(results)[:12]


def generate_answer(question, refs, corpus, names, client, model, explicit):
    evidence = []
    for ref in refs[:7]:
        k, n = ref["kind"], ref["article"]
        evidence.append(render_article(names[k][0], n, corpus[k][n])[:11500])
    system = """あなたは地方自治法等を説明する法令検索AIです。提示されたe-Gov原文のみを根拠に答える。
法律上の『承認を求める』と『承認を得る』を混同しない。
第167条（指名競争入札）と第167条の2（随意契約）を混同しない。
第167条の2第1項第1号の金額基準を第5号などに一律適用しない。
条例・自治体規則・会議規則・長の定め等に委ねられる事項は、その具体的内容を断定しない。
引用先の他の国の法律・政令・省令の本文が提示されていなければ、その内容を確認済みとしない。
未施行改正・判例・通知・自治体例規を推測で補わない。
引用は提示した法令識別子・条番号に限定し、実際に根拠として使った条文だけを挙げる。
JSONのみで回答する。形式:
{"conclusion":"...","explanation":"...","conditions":["..."],"cautions":["..."],"citations":[{"kind":"令","article":"167_2","paragraph":"1","item":"5"}]}"""
    prompt = "識別子: " + ", ".join(f"{k}={v[0]}" for k, v in names.items())
    prompt += "\n質問: " + question + "\n\n確認済み条文:\n" + "\n\n".join(evidence)[:36000]
    answer = ai_json(client, model, system, prompt, 2600)
    if not isinstance(answer, dict):
        raise ValueError("回答の形式が不正です")
    citations = []
    for c in answer.get("citations", []):
        if not isinstance(c, dict):
            continue
        k, n = c.get("kind"), normalize_num(c.get("article"))
        if not any(r["kind"] == k and r["article"] == n for r in refs[:7]):
            continue
        art = corpus[k][n]
        p = str(c.get("paragraph") or "")
        item = str(c.get("item") or "")
        if p and p not in {x["num"] for x in art["paragraphs"]}:
            continue
        if item and not any(i["num"] == item for x in art["paragraphs"] if not p or x["num"] == p for i in x["items"]):
            continue
        citation = {"kind": k, "article": n, "paragraph": p, "item": item}
        if citation not in citations:
            citations.append(citation)
    answer["citations"] = citations
    errors = []
    text = " ".join(str(answer.get(x, "")) for x in ("conclusion", "explanation", "conditions", "cautions"))
    if not citations:
        errors.append("回答の根拠条文を検証できませんでした。")
    if any(c["kind"] == "法" and c["article"] == "179" for c in citations):
        if re.search(r"承認を(?:必ず)?得なければならな|承認を得る必要があ|承認を受けなければならな", text):
            errors.append("第179条の『承認を求める』と『承認を得る』の混同を検出しました。")
    if "出張" in question and re.search(r"必ず.{0,12}(代理|副市長)|出張.{0,12}当然.{0,12}代理", text):
        errors.append("出張のみを理由とする当然の職務代理という断定を検出しました。")
    return answer, errors


with st.sidebar:
    st.header("検索設定")
    chosen = st.multiselect("検索する基本法令", list(LAWS), default=list(LAWS),
                            format_func=lambda k: LAWS[k][0])
    st.caption("自治体の条例・規則等は取得しません。確認が必要な場合に注意を表示します。")
    st.markdown("[e-Gov法令API](https://laws.e-gov.go.jp/api/)")
    with st.expander("他の国の法令を追加（任意）"):
        other_name = st.text_input("法令名", placeholder="例：行政手続法")
        other_id = st.text_input("e-Gov法令ID", placeholder="例：405AC0000000088")
        st.caption("法令名だけで自動特定はしません。正しい法令IDをe-Govで確認して入力してください。")
    model = st.text_input("Geminiモデル", value=DEFAULT_MODEL)

question = st.text_area("地方自治法等について質問してください", height=110,
                        placeholder="例：少額随意契約の金額基準は自治体の規則で定めますか？")
run = st.button("条文を検索して回答", type="primary", disabled=not chosen and not other_id.strip())

if run:
    if not question.strip():
        st.warning("質問を入力してください。")
        st.stop()
    names = {k: LAWS[k] for k in chosen}
    if other_id.strip() or other_name.strip():
        if not (other_id.strip() and other_name.strip()):
            st.error("追加法令は法令名と法令IDを両方入力してください。")
            st.stop()
        if not re.fullmatch(r"[0-9A-Z]{15}", other_id.strip().upper()):
            st.error("法令IDの形式を確認してください（英数字15文字）。")
            st.stop()
        names["追加"] = (other_name.strip(), other_id.strip().upper())
    try:
        with st.spinner("e-Govから法令原文を取得しています..."):
            corpus = {k: fetch_law(law_id) for k, (_, law_id) in names.items()}
        client = client_for_gemini()
        with st.spinner("根拠条文を検索しています..."):
            refs, explicit = find_candidates(question, corpus, names, client, model)
        if not refs:
            st.warning("根拠条文を特定できませんでした。改正履歴など、条文検索以外の質問には対応できない場合があります。")
            st.stop()
        with st.spinner("根拠条文から回答を作成しています..."):
            answer, errors = generate_answer(question, refs, corpus, names, client, model, explicit)

        # 重要な確認事項は回答より前に、大きく・明確に表示する。
        # 回答の根拠として引用された項・号に絞り、別の論点の警告を混ぜにくくする。
        all_checks = local_checks(refs[:7], corpus, names)
        cited_checks = []
        for check in all_checks:
            for cite in answer.get("citations", []):
                prefix = f"{names[cite['kind']][0]} {article_label(cite['article'])}"
                if not check["source"].startswith(prefix + "第"):
                    continue
                if cite.get("paragraph") and f"第{cite['paragraph']}項" not in check["source"]:
                    continue
                if cite.get("item") and f"第{cite['item']}号" not in check["source"]:
                    continue
                cited_checks.append(check)
                break
        if cited_checks:
            st.error("🚨【重要】この回答には、自治体の条例・規則等の確認が必要な事項があります。国の法令だけで最終判断しないでください。", icon="🚨")
            st.markdown("### 🏛️ 必ず当該自治体で確認してください")
            for check in cited_checks[:5]:
                st.markdown(f"**● {check['type']}を確認：{check['source']}**")
            st.markdown("**確認先：当該自治体の公式例規集・契約担当部署など**")
            st.caption("このアプリは各自治体の条例・規則等を取得・検証していません。")
        elif all_checks:
            st.warning("⚠️【自治体例規の確認】検索候補の条文に条例・規則等への委任表現があります。今回の回答との関係は未確定です。下の詳細を確認してください。")

        st.subheader("自治体の条例・規則等の確認")
        checks = cited_checks if cited_checks else all_checks
        if checks:
            st.warning("⚠️ 自治体の現行の条例・規則等を確認してください。上部の重要警告は、回答に引用された条項を優先しています。")
            st.caption("このアプリは自治体例規を自動取得・検証しません。以下は機械的な候補抽出であり、網羅性や委任関係の確定を保証しません。")
            for c in checks[:15]:
                st.write(f"・**{c['source']}**：{c['type']}の確認が必要")
                with st.expander("検出した条文の記載を確認"):
                    st.text(c["excerpt"])
        else:
            st.info("検索した条文から自治体例規への明示的な委任表現は検出されませんでした。ただし、自治体例規の確認が不要と確定したわけではありません。")
        st.caption("金額、要件、手続など自治体固有の事項は、当該自治体の担当部署や公式例規集で確認してください。")


        st.subheader("条文検索結果")
        search_text = "\n".join(
            f"・{names[r['kind']][0]} {article_label(r['article'])}"
            for r in refs[:7]
        )
        st.write(search_text)
        st.code(search_text, language=None)
        st.caption("上の枠のコピーアイコンで条文検索結果をコピーできます。")

        st.subheader("AIの回答")
        answer_lines = []
        if errors:
            st.error("回答の検証で問題を検出しました。解釈文は表示せず、根拠原文を確認してください。")
            answer_lines.append("【検証で問題を検出：AIの解釈文は非表示】")
            for e in errors:
                st.write("・" + e)
                answer_lines.append("・" + e)
        else:
            for title, key in (("結論", "conclusion"), ("説明", "explanation")):
                value = str(answer.get(key, ""))
                st.markdown(f"**{title}**")
                st.write(value)
                answer_lines.extend([f"【{title}】", value, ""])
            for title, key in (("適用条件", "conditions"), ("注意点", "cautions")):
                values = answer.get(key, [])
                if isinstance(values, list) and values:
                    st.markdown(f"**{title}**")
                    answer_lines.append(f"【{title}】")
                    for v in values:
                        st.write("・" + str(v))
                        answer_lines.append("・" + str(v))
                    answer_lines.append("")
        if answer["citations"]:
            st.markdown("**回答に用いた根拠条文**")
            answer_lines.append("【回答に用いた根拠条文】")
            for c in answer["citations"]:
                suffix = (f"第{c['paragraph']}項" if c["paragraph"] else "") + (f"第{c['item']}号" if c["item"] else "")
                citation_line = f"・{names[c['kind']][0]} {article_label(c['article'])}{suffix}"
                st.write(citation_line)
                answer_lines.append(citation_line)
        answer_copy = "\n".join(answer_lines).strip()
        st.markdown("**AIの回答をコピー**")
        st.code(answer_copy, language=None)
        st.caption("上の枠のコピーアイコンでAIの回答をまとめてコピーできます。")

        # 引用された条文を最優先に表示する。AI回答が抑止されても原文は表示する。
        st.subheader("e-Govから取得した根拠条文原文")
        st.caption("AIの言い換えではなくe-Gov XML由来の本文です。附則や別表はこの版では対象外です。")
        priority = []
        for c in answer["citations"]:
            key = (c["kind"], c["article"])
            if key not in priority:
                priority.append(key)
        for r in refs[:7]:
            key = (r["kind"], r["article"])
            if key not in priority:
                priority.append(key)
        original_texts = [
            render_article(names[k][0], n, corpus[k][n])
            for k, n in priority
        ]
        st.markdown("**根拠条文原文をまとめてコピー**")
        st.code("\n\n".join(original_texts), language=None)
        st.caption("上の枠のコピーアイコンで表示対象の条文原文を一括コピーできます。")
        for index, (k, n) in enumerate(priority):
            citation = next((c for c in answer["citations"] if c["kind"] == k and c["article"] == n), None)
            heading = f"{names[k][0]} {article_label(n)}"
            with st.expander(heading, expanded=(index == 0 or (citation is not None and index < 3))):
                st.text(render_article(names[k][0], n, corpus[k][n]))
                st.link_button("e-Govで法令を確認", f"https://laws.e-gov.go.jp/law/{names[k][1]}")

        st.subheader("他の国の法令への引用候補")
        mentions = national_mentions(refs[:7], corpus, names)
        if mentions:
            st.caption("以下は条文から抽出した文字列の候補です。引用先法令の同定・本文取得・適用関係の確認は未実施です。")
            for m in mentions:
                st.write("・" + m)
            st.info("引用先の国の法令を根拠に加える場合は、サイドバーで正式な法令名とe-Gov法令IDを入力し、再検索してください。")
        else:
            st.caption("引用候補は検出されませんでした。引用関係がないことを意味しません。")

        st.subheader("施行状況に関する注意")
        st.info("API Version 1から取得した本文を表示しています。未施行改正の条項別判定は行っていません。")
        for k, (_, law_id) in names.items():
            dates, ok = revision_status(law_id)
            if dates:
                st.warning(f"{names[k][0]}：履歴から将来の施行日候補を検出しました（{', '.join(dates[:4])}）。該当条項は未判定です。")
            elif not ok:
                st.caption(f"{names[k][0]}：改正履歴を取得できませんでした。")
            else:
                st.caption(f"{names[k][0]}：取得した履歴から将来の施行日を検出しませんでした。未施行改正が存在しないことの証明ではありません。")
    except Exception as exc:
        st.error("取得または回答処理に失敗しました。")
        st.code(str(exc))
