Basic Usage of Facebook Group Analysis
======================================

We have a nice JSON of all the messages from a group chat we're in.
It'd be nice to analyze the data stored in the file. However, writing
all the requisite methods is time consuming. Therefore, we'll show how using the
library can greatly expedite the process.


Getting Started
===============

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
opposed to saving the output of :py:meth:`~fgroupanalysis.raw_text` the text is fixed up to a readable standard (i.e you can see emojis)