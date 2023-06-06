import sys
import re
from ast import literal_eval


line1 = []
line2 = []

dummy = []

line1.append('Er sagte: "Jimmy, was machen Sie im Gerichtssaal?" Der ältere schwarze Mann stand auf. Er sah den Deputy an, dann sah er mich an, und er sagte: "Ich bin hergekommen, um diesem jungen Mann zu sagen, Verlieren Sie das Ziel nicht aus den Augen. Geben Sie nicht auf. ‌"')
line2.append('Er sagte: "Jimmy, was machen Sie im Gerichtssaal?" Der ältere schwarze Mann stand auf. Er sah den Deputy an, dann sah er mich an, und er sagte: "Ich bin hergekommen, um diesem jungen Mann zu sagen, Verlieren Sie das Ziel nicht aus den Augen. Geben Sie nicht auf. "')
line3.append('Und ich fing an mit diesem Antrag. Die Überschrift lautete: » Antrag, meinen armen, 14jährigen schwarzen Mandanten wie einen privilegierten, weißen, 75jährigen Topmanager zu behandeln. ‌ «')
line4.append('Und ich fing an mit diesem Antrag. Die Überschrift lautete: » Antrag, meinen armen, 14jährigen schwarzen Mandanten wie einen privilegierten, weißen, 75jährigen Topmanager zu behandeln. «')
#print(repr(line1[0]))


if line1[0].replace("‌", "")== line2[0]:

    print("equal")

else:
    print('not equal')
    