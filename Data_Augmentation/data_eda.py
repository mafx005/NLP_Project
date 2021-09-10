# -*- coding: utf-8 -*-
# @Time : 2021/8/20 14:04
# @Author : M
# @FileName: data_eda.py
# @Dec : 
import jieba
import synonyms
import random
from random import shuffle


"""
paper title:    Easy Data Augmentation Techniques for Boosting Performance on TextClassification Tasks
"""

random.seed(42)
# 加载自定义词典，根据任务需要自己修改
jieba.load_userdict('dict/dict.txt')

# 停用词列表，默认使用哈工大停用词表
f = open('stopwords/hit_stopwords.txt', 'r', encoding='utf8')
stop_words = list()
for stop_word in f.readlines():
    stop_words.append(stop_word[:-1])


def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))  # 过滤停用词
    # random_word_list = list(set([word for word in words]))    # 不过滤停用词
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    return synonyms.nearby(word)[0]


def random_insertion(words, n):
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return new_words


def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def random_swap(words, n):
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def random_deletion(words, p):
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def nlp_eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
    seg_list = jieba.cut(sentence)
    seg_list = " ".join(seg_list)
    words = list(seg_list.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(''.join(a_words))

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_words = random_insertion(words, n_ri)
        augmented_sentences.append(''.join(a_words))

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(''.join(a_words))

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(''.join(a_words))

    # print(augmented_sentences)
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]
    seg_list = ''.join(seg_list.split())
    augmented_sentences.append(seg_list)
    # print(len(augmented_sentences))
    # augmented_sentences_set = list(set(augmented_sentences))
    # print(len(augmented_sentences_set))

    return augmented_sentences


def gen_eda(train_orig, output_file, alpha, num_aug=9, flag=False):
    writer = open(output_file, 'w', encoding='utf8')
    lines = open(train_orig, 'r', encoding='utf8').readlines()
    # lines = lines[:30]

    print("正在使用EDA生成增强语句...")
    for i, line in enumerate(lines):
        if i % 5000 == 0:
            print(i)
        # print(i)
        if flag:
            parts = line.strip().split('\t')
            assert len(parts) == 2
            label = parts[0]
            sentence = parts[1]
            prefix = label + "\t"
        else:
            sentence = line.strip()
            prefix = ''
        aug_sentences = nlp_eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            writer.write(prefix + aug_sentence + '\n')

    writer.close()
    print("已生成增强语句!")
    print(output_file)


if __name__ == '__main__':
    num_aug = 3  # 语句增强倍数，需要生成几倍的数据
    alpha = 0.2  # 语句中被替换、删除、插入的单词数占比
    flag = True    # 数据是否存在label，如果存在,按照label \t sentence格式保存,并将flag修改为True
    input_filename = r'D:\CODE\cail_2021\data\train_0812_labels_clean.txt'  # 输入要增强的文件
    output_filename = r'D:\CODE\cail_2021\data\train_0812_labels_clean_augment.txt'  # 输出增强后的文件
    gen_eda(input_filename, output_filename, alpha=alpha, num_aug=num_aug, flag=flag)
