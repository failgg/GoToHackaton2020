import os
from io import BytesIO

import matplotlib.image as mpimg
import telebot
import Smartload
from telebot import types
from telebot import apihelper

token = ''
apihelper.proxy = {'https': 'socks5h://geek:socks@t.geekclass.ru:7777'}
bot = telebot.TeleBot(token=token, threaded=False)


def handle_text(message):
    return message


@bot.message_handler(commands=["start"])
def repeat_all_messages(message):
    keyboard = types.InlineKeyboardMarkup()
    button1 = types.InlineKeyboardButton(text="FAQ", callback_data="help")
    button2 = types.InlineKeyboardButton(text="Download photo", callback_data="send_image")
    button3 = types.InlineKeyboardButton(text="Recognize", callback_data="recognize_image")
    keyboard.add(button1)
    keyboard.add(button2)
    keyboard.add(button3)
    bot.send_message(message.chat.id, "Use commands", reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.message:
        if call.data == "help":
            bot.send_message(call.message.chat.id, "Recognize Bakugans")
            file_info = bot.get_file(handle_add_photo.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            bot.send_message(call.message.chat.id, "got")
        elif call.data == "send_image":
            msg = bot.send_message(call.message.chat.id, "Send me a photo")
            bot.register_next_step_handler(callback=handle_add_photo, message=msg)
        elif call.data == "recognize_image":
            bot.send_message(call.message.chat.id, "already")


# @bot.message_handler(content_types=['photo'])
# def parse_all_photos(message):
#   f = open('data/img/test.png', 'rb')
#  res = bot.send_photo(message.chat.id, f, None)
# print(message.photo[-1].file_id)
@bot.message_handler(content_types=['photo'])
def handle_add_photo(message):
    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    ind = Smartload.predict(BytesIO(downloaded_file))
    # dirr = str(ind)+'.txt'
    # file=open(dirr,'r')
    # text=file.read()
    # bot.send_message(message.chat.id, text)
    if ind == 0:
        bot.send_message(message.chat.id, "Centipoid")
    elif ind == 1:
        bot.send_message(message.chat.id, "Blade Tigrerra")

    # bot.send_photo(message.chat.id,downloaded_file)


bot.polling(none_stop=True)
