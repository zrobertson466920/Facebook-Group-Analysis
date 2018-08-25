import fgroupanalysis as fg
from fgroupanalysis import Timeline

def main():

    # Open and Read File
    file = open('message.json','r')
    raw = file.read()
    timeline = Timeline(raw)

    # Splits the Messages By Sender
    timelines = timeline.partition()

    # Flatten Structure into List of Messages
    text = timeline.raw_text()

    fg.activity_plot(timeline.get_timestamps(),0)

main()
