# Will fix unicode errors present in the file formatting
import numpy as np
import matplotlib.pyplot as plt
from ftfy import fix_text
import json

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import hdbscan
import matplotlib.pyplot as plt

import datetime

from collections import Counter

from wordcloud import WordCloud

class Timeline:
    def __init__(self,raw = None):
        if raw == None:
            self.messages = []
            return
        else:
            data = json.loads(raw)
            timeline = data['messages']
            timeline.reverse()
            self.messages = timeline
            return

    # Splits the messages by sender
    def partition(self):
        actors = []
        timelines = {}
        for message in self.messages:
            if message['sender_name'] in actors:
                timelines[message['sender_name']].messages.append(message)
            else:
                actors.append(message['sender_name'])
                timelines[message['sender_name']] = Timeline()
                timelines[message['sender_name']].messages.append(message)
        return timelines

    # Fixes encoding errors
    def to_text(self,file):
        f = open(file,'w')
        for message in timeline.messages:
            try:
                f.write(fix_text(message['content']) + "\n")
            except:
                continue
        f.close()

    # Constructs a dictionary of edge frequencies where edges are (sender,tagged)
    def tag_matrix(self,num = None):
        t_dict = {}
        tags = []
        for message in self.messages:
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
    def reac_matrix(self,num = None):
        reactions = []
        for message in self.messages:
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

    # Creates a list of time gaps between a person's message to the group and the first reply back
    def reply_gap(self,person):
        reply_gap = []
        for i in range(len(self.messages)-1):
            if self.messages[i]['sender_name'] == person and self.messages[i+1]['sender_name'] != person:
                reply_gap.append(self.messages[i+1]['timestamp']-self.messages[i]['timestamp'])
        return reply_gap

    def get_timestamps(self):
        timestamps = []
        for message in self.messages:
            timestamps.append(message['timestamp'])
        return timestamps

    def raw_text(self):
        text = []
        for message in self.messages:
            try:
                text += "\n" + message['content']
            except:
                continue
        return text



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

# Plots the number of messages sent in each hour slot during the day
# time_difference (hours) will account for timezone differences
# aka time_difference = 1 assumes you're one hour ahead
def activity_plot(timestamps,time_difference = 0):
    times = []
    for stamp in timestamps:
        temp = datetime.datetime.fromtimestamp(stamp)
        times.append((temp.hour+temp.minute / 60 - time_difference) % 24)
    return times

# Groups Timestamps into Clusters
# (5-10) ~ are good numbers for min_cluster
def cluster(timestamps,min_cluster):

    X = np.array(timestamps)
    X = X.reshape(-1,1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size= min_cluster, min_samples = 2)
    labels = clusterer.fit_predict(X)

    count = (len(set(labels)) - (1 if -1 in labels else 0))

    #Number of clusters in labels, ignoring noise if present
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    #Black is removed and used for noise
    unique_lables = set(labels)

    cmap = plt.cm.get_cmap("Spectral")
    colors = [cmap(np.random.rand(1)[0]) for each in np.linspace(0, 1, len(unique_lables))]

    for k, col in zip(unique_lables, colors):
        if k == -1:
            # Black used for noise
            col = [0, 0, 0, 1]
            continue

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
        plt.plot(xy[:], [0] * len(xy), 'o', c=tuple(col), linewidth = 7)

    plt.title("Estimated Number of Clusters: %d" % n_clusters_)
    plt.show()

    return labels
