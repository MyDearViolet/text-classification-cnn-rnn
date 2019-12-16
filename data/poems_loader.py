# coding: utf-8
import sys
from collections import Counter
import numpy as np
import tensorflow.keras as kr
import json
import re

PoemTypes = []
with open('poems/poems-types.txt', 'r') as f:
    # 逐行解析json
    for line in f:
        PoemTypes.append(line.strip())

PoemsData = []
PoemsList = ['poems1.json', 'poems2.json', 'poems3.json', 'poems4.json']
for filename in PoemsList:
    with open('poems-db/'+filename, 'r') as f:
        # 逐行解析json
        for line in f:
            line_json = json.loads(line)
            tags = [tag for tag in line_json["tags"] if tag in PoemTypes]
            if len(tags) > 0 and line_json["content"] is not None:
                # 读取每首诗的标签
                str_tags = ''
                for tag in tags:
                    str_tags += tag + ','
                # 读取每首诗的内容
                str_poem = ''
                for content in line_json["content"]:
                    str_poem += content
                # 去除“<p>”、“()”、“\n”等字符串
                str_poem = re.sub('[\(\（\<][\S\s]*?[\)\）\>]', '', str_poem, 0)
                str_poem = re.sub('\s', '', str_poem, 0)
                # 去除过长诗词
                if len(str_poem) > 80:
                    continue
                else:
                    PoemsData.append([str_tags.strip(','), str_poem.strip()])

# with open('poems/poems.train.txt', 'w+') as f:
#     for line in PoemsData:
#         str_content = ''
#         for content in line:
#             str_content += content + ' '
#         f.write(str_content.strip())

np.savetxt('poems/poems.train.txt', PoemsData, '%s')
