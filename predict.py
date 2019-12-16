# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
                 'Apple公司最近发布了新一代iPhone手机，新iPhone拥有更优秀的处理速度、更好的成像画质和更长的续航',
                 '今天股市大跌，大批量股票出现断崖式下跌，众股民纷纷抛售股票以期减少损失',
                 '近日，浙江大学可视化分析小组发布新型大气可视化分析系统，可用于将复杂的气象数据用易懂的方式呈现出来',
                 '床前明月光，疑似地上霜。',
                 '国破山河在，城春草木深。',
                 '近日，著名歌手蔡徐坤发布最新专辑《鸡你太美》，引发网友模仿热潮',
                 '近日，国家主席习近平对美国进行国事访问，美国总统特朗普热情接待并邀请习总书记共进晚餐']
    for i in test_demo:
        print(cnn_model.predict(i))
