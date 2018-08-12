# Will fix unicode errors present in the file formatting
import numpy as np
import matplotlib.pyplot as plt
from ftfy import fix_text
import json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

import datetime

from collections import Counter

from wordcloud import WordCloud

# Fixes encoding errors
def to_text(timeline,file):
    f = open(file,'w')
    for message in timeline:
        try:
            f.write(fix_text(message['content']) + "\n")
        except:
            continue
    f.close()

# Splits the messages by sender
def partition(timeline):
    actors = []
    timelines = {}
    for message in timeline:
        if message['sender_name'] in actors:
            timelines[message['sender_name']].append(message)
        else:
            actors.append(message['sender_name'])
            timelines[message['sender_name']] = []
            timelines[message['sender_name']].append(message)
    return timelines

# Will return the top 'num' words as a list of tuples [(word,num)]
def common_words(text,num,pl = -1):
    #From NTLK library
    stop_words = set(stopwords.words("english"))

    # Fix encoding on text and then format to lowercase
    word_tokens = word_tokenize(fix_text(text).lower())

    # Gets rid of contractions in addition to stopwords
    filtered_words = [w for w in word_tokens if ((not w in stop_words) and (len(w) > 3))]

    freq = Counter(filtered_words)

    # Show a plot/list or nothing depending on pl
    if pl == -1:
        return freq
    elif pl == 0:
        sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
        for key,val in sorted_by_second[0:num]:
            print(str(key) + ":" + str(val))
    else:
        wordcloud = WordCloud(width=1600, height=800).generate_from_frequencies(freq)
        plt.figure(figsize = (20,10), facecolor = 'k')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

    return freq

# Constructs a frequnecy list based off of tag_matrix
def tag_freq(freq_tuples,num,pl = -1):

    freq = []
    for (tagger,tagged),count in freq_tuples:
        freq += [tagged] * count
    freq = Counter(freq)

    if pl == -1:
        return freq
    elif pl == 0:
        sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
        for key,val in sorted_by_second[0:num]:
            print(str(key) + ":" + str(val))
    else:
        sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
        freq = sorted_by_second[:num]
        plt.bar(range(len(freq)), [val[1] for val in freq], align='center')
        plt.xticks(range(len(freq)), [val[0] for val in freq])
        plt.xticks(rotation=70)
        plt.show()

def reac_freq(reac_tuples,num,pl = -1):

    freq = []
    for (tagger,reaction,tagged),count in reac_tuples:
        freq += [tagged] * count
    freq = Counter(freq)

    if pl == -1:
        return freq
    elif pl == 0:
        sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
        for key,val in sorted_by_second[0:num]:
            print(str(key) + ":" + str(val))
    else:
        sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
        freq = sorted_by_second[:num]
        plt.bar(range(len(freq)), [val[1] for val in freq], align='center')
        plt.xticks(range(len(freq)), [val[0] for val in freq])
        plt.xticks(rotation=70)
        plt.show()


# Constructs a dictionary of edge frequencies where edges are (sender,tagged)
def tag_matrix(timeline,num = None):
    t_dict = {}
    tags = []
    for message in timeline:
        try:
            word_tokens = word_tokenize(fix_text(message['content']).lower())
            tag_indices = [i for i, tag in enumerate(word_tokens) if tag == '@']
            for i in tag_indices:
                tags.append((message['sender_name'],word_tokens[i] + word_tokens[i+1]))
        except:
            continue
    freq = Counter(tags)
    sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
    freq = sorted_by_second[:num]
    print(freq)
    return freq

# Constructs a dictionary of reaction frequencies where edges (reaction wheighted) are (sender,reaction,receiver)
def reac_matrix(timeline,num = None):
    reactions = []
    for message in timeline:
        try:
            for reac in message['reactions']:
                reactions.append((reac['actor'],fix_text(reac['reaction']),message['sender_name']))
        except:
            continue
    freq = Counter(reactions)
    sorted_by_second = sorted(freq.items(), key=lambda tup: tup[1], reverse = True)
    freq = sorted_by_second[:num]
    print(freq)
    return freq

# Plots the number of messages sent in each hour slot during the day
def activity_plot(timestamps):
    times = []
    for stamp in timestamps:
        temp = datetime.datetime.fromtimestamp(stamp)
        times.append((temp.hour+temp.minute / 60 - 1) % 24)
    plt.hist(times,24)
    plt.show()

# Creates a list of time gaps between a person's message to the group and the first reply back
def reply_gap(timeline,person):
    reply_gap = []
    for i in range(len(timeline)-1):
        if timeline[i]['sender_name'] == person and timeline[i+1]['sender_name'] != person:
            reply_gap.append(timeline[i+1]['timestamp']-timeline[i]['timestamp'])
    return reply_gap

def main():

    # Open and Read File
    file = open('message.json','r')
    raw = file.read()
    data = json.loads(raw)
    timeline = data['messages']
    timeline.reverse()

    # Splits the Messages By Sender
    timelines = partition(timeline)

    #Store Timestamp Data
    timestamps = []
    text = ''
    f = open("timestamps.txt",'w')
    for message in timeline:
        timestamps.append(message['timestamp'])
        f.write(str(message['timestamp']) + '\n')
        try:
            text += "\n" + message['content']
        except:
            continue
    f.close()

    #Histograms
    #freq_words = common_words(text,15,1)
    #tag_freq(tag_matrix(timelines['Zach Robertson']),10,1)
    #tag_matrix(timeline,10)
    #reac_freq(reac_matrix(timeline),10,1)
    #reac_matrix(timeline,10)
    #activity_plot(timestamps)

    time_gaps = reply_gap(timeline,'Dave Marques')
    time_gaps = [tg for tg in time_gaps if tg < 100]
    plt.hist(time_gaps,100)
    plt.show()

main()
