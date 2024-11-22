import pandas as pd
from difflib import SequenceMatcher

# 加载数据
file_path = "实验1数据.xlsx"
sheet1 = pd.read_excel(file_path, sheet_name='sheet1', header=None, names=['Index', 'Journal Name', 'Impact Factor'])
sheet2 = pd.read_excel(file_path, sheet_name='sheet2', header=None,
                       names=['Journal Name', 'Abbreviation1', 'Abbreviation2', 'Abbreviation3', 'Abbreviation4'])


# 数据预处理
def normalize_name(name):
    """统一期刊名称格式，去除多余空格和符号，替换标点等"""
    if pd.isna(name):
        return ""
    name = name.strip().replace('&', 'and').replace('-', ' ').replace(',', '')
    return name.lower()


# 预处理期刊名称
sheet1['Normalized Name'] = sheet1['Journal Name'].apply(normalize_name)
sheet2['Normalized Name'] = sheet2['Journal Name'].apply(normalize_name)


# 匹配期刊名
def find_best_match(name, candidates):
    """根据相似度找到最佳匹配"""
    best_match = None
    highest_ratio = 0.8  # 设定相似度阈值
    for candidate in candidates:
        ratio = SequenceMatcher(None, name, candidate).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = candidate
    return best_match, highest_ratio


# 创建结果数据框
result = []

sheet2_names = sheet2['Normalized Name'].tolist()

for _, row in sheet1.iterrows():
    original_name = row['Journal Name']
    normalized_name = row['Normalized Name']
    impact_factor = row['Impact Factor']

    # 在 sheet2 中寻找最佳匹配
    best_match, similarity = find_best_match(normalized_name, sheet2_names)
    if best_match:
        abbreviations = sheet2[sheet2['Normalized Name'] == best_match].iloc[0, 1:].dropna().tolist()
    else:
        abbreviations = []

    result.append([original_name, impact_factor] + abbreviations)

# 转换为 DataFrame 并排序
result_df = pd.DataFrame(result,
                         columns=['Journal Name', 'Impact Factor', 'Abbreviation1', 'Abbreviation2', 'Abbreviation3', 'Abbreviation4'])
result_df = result_df.sort_values(by='Journal Name')

# 保存结果到 Sheet3
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    result_df.to_excel(writer, sheet_name='Sheet3', index=False)

print("结果已保存至 Sheet3。")
