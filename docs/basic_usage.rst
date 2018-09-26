Basic Usage of Facebook Group Analysis
======================================

We have a nice JSON of all the messages from a group chat we're in.
It'd be nice to analyze the data stored in the file. However, writing
all the requisite methods is time consuming. Therefore, we'll show how using the
library can greatly expedite the process.


Getting Started
---------------

So we have the JSON, let's go ahead and start processing it.

.. code:: python

    # Open and Read File
    timeline = Timeline(raw = 'message.json')

In this snippet, we're opening the JSON file, reading the data directly into a class object
called :py:meth:`~fgroupanalysis.Timeline` and then returning the object to the user. Now,
one of the main things we'll want access to are the messages and the participants. We can get access to these using,

.. code:: python

    participants = timeline.participants
    messages = timeline.messages

It's also fairly common to just want a raw text file containing all of the messages disregarding meta-data,

.. code:: python

    text = timeline.raw_text()

The text string is newline separated. Additionally, if you'd like to just directly save the text to a file there's
also,

.. code:: python

    timeline.to_text(file_name)

Why was this method made? The reason is because character conversion is not 100% accurate. When you use this method as
opposed to saving the output of :py:meth:`~fgroupanalysis.Timeline.raw_text` the text is fixed up to a readable standard (i.e you can see emojis)


Phrase Analysis
---------------

We can find the most commonly used words in a group chat using,

.. code:: python

    word_freq = fg.common_words(text, plot = False)

This method will return a list of tuples of the form (word,frequency). If you want to see a simple word cloud of
the output make the plot variable true. Similarly, we can collect the most reacted messages using,

.. code:: python

    reacted_messages = timeline.message_reacs()

Where the return is now a list of tuples of the form (sender,content,reac_count). Another important feature is the ability to partition, combine, or filter
the :py:meth:`~fgroupanalysis.Timeline` objects. For example, we can split a timeline into a dictionary sorted by sender:

.. code:: python

    timelines = timeline.partition()

We can also recombine those messages using,

.. code:: python

    timeline = combine(timelines.values())

Moreover, it's also possible to filter a timeline based on a desired criteria. For example, if we want to filter the timeline by mentions of the
phrase 'anyone' we can use,

.. code:: python

    anyone_timeline = timeline.filter_by_word('anyone')

Interaction Analysis
--------------------

We can analyze user interaction based on time, reaction, or tagging behavior. For instance, we can count how many times user B follows user B using,

.. code:: python

    markov_edges = timeline.markov_edges()

Where we're returning a list of tuples of the form ((user A, user B), num). We can perform a similar analysis of tagging behavior with,

.. code:: python

    tag_matrix = timeline.tag_matrix()

Finally, we can calculate a similar list for reaction counts using,

.. code:: python

    reac_matrix = timeline.reac_matrix()

Where the tuples are now of the form ((sender,reaction,receiver), count)