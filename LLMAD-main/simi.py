import spacy
import re
from collections import Counter

# 加载英文模型
nlp = spacy.load("en_core_web_sm")

# 原始数据
data ={
  "* Quasi-periodic oscillation with consistent cycle length (typically 10–14 time steps per cycle, slightly longer than class 3) and low period jitter; start-up may show a single deep negative transient before settling, producing regular repeated pulses across the series.":"",
"* Asymmetric pulses: narrow, sharp positive peaks with larger magnitude and shorter duration than the wider, smoother negative troughs (positive skew). Crests are typically two-stage: a small, consistent shoulder pre-peak precedes the main spike, producing a consistently multi-modal peak shape (doublet-like) within each cycle. Negative troughs are deeper than positive crests (typical trough-to-crest amplitude ratio >1.3). The positive phase is brief (<20–30% of the cycle) with a steep upstroke and rapid decay, while the negative phase is longer and smoother (broader trough).":"",
"* Axis-dependent baseline offset: cycles often sit slightly below zero on this axis (negative DC bias), though other axes may show near-zero or mild positive bias.":"",
"* A repeating beat-like alternation every 2–3 cycles; unlike class 3’s smooth envelope drift, the alternation is discrete and pattern-like.":"",
"partial-sample feature":"",
"* Occasional isolated deep negative transients (outlier troughs) that are much deeper than typical troughs and momentarily break the regular pattern.":"",
"* Subtle intra-cycle micro-notches/jitter appear mainly within troughs; late segments may gain jitter but cycles retain the two-stage crest pattern.":"",
}


# 合并所有键为一段话
combined_text = " ".join(data.keys())

# === 数字识别（增强版）===
# 匹配：±, ≈, −（Unicode minus）, en-dash –, hyphen -, 小数，整数，范围
digit_pattern = re.compile(
    r'[±≈]?\s*'                    # 可选符号
    r'(?:−|\-)?\d+(?:\.\d+)?'      # 可能带负号的数字（支持 Unicode minus '−'）
    r'(?:\s*[–\-]\s*'              # 可选范围连接符
    r'(?:−|\-)?\d+(?:\.\d+)?)?'    # 范围的第二部分（可选）
)
# 查找所有数字表达式（去重不是必须，这里保留全部出现）
number_matches = digit_pattern.findall(combined_text)
# 修复：上面正则分组可能导致只返回部分，改用 finditer 获取完整匹配
number_expressions = [match.group() for match in digit_pattern.finditer(combined_text)]

# === spaCy 处理 ===
doc = nlp(combined_text)

# 词元过滤（用于词性统计）
tokens = [token for token in doc if not token.is_space and not token.is_punct]

# 名词 & 形容词（唯一，lemma 小写）
nouns = {token.lemma_.lower() for token in tokens if token.pos_ == "NOUN"}
adjs = {token.lemma_.lower() for token in tokens if token.pos_ == "ADJ"}

# 命名实体识别
entities = [(ent.text, ent.label_) for ent in doc.ents]
entity_counts = Counter([ent.label_ for ent in doc.ents])

# === 输出结果 ===
print("=== 合并后的段落 ===")
print(combined_text)
print("\n" + "="*60)

print(f"字符长度（含空格）: {len(combined_text)}")
print(f"词数（不含标点/空格）: {len(tokens)}")
print(f"唯一名词数量: {len(nouns)} → {sorted(nouns)}")
print(f"唯一形容词数量: {len(adjs)} → {sorted(adjs)}")
print(f"数字表达式数量: {len(number_expressions)} → {number_expressions}")

print(f"\n命名实体总数: {len(entities)}")
if entities:
    print("识别出的实体（文本 → 类型）:")
    for text, label in entities:
        print(f"  '{text}' → {label} ({spacy.explain(label)})")
    print("\n按类型统计:")
    for ent_type, count in entity_counts.most_common():
        print(f"  {ent_type} ({spacy.explain(ent_type)}): {count}")
else:
    print("未识别出任何命名实体。")