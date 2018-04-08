# -*- coding: utf-8 -*-

from chatterbot import ChatBot
import data_utils

# Create a new chat bot named Charlie
chatbot = ChatBot(
    'Charlie',
    trainer='chatterbot.trainers.ListTrainer',
    read_only=True
)

# conv_list = data_utils.load_xhj()
# for a_list in conv_list:
#     chatbot.train(a_list)

# Get a response to the input text 'How are you?'
# while True:
print(chatbot.get_response("i love you"))
