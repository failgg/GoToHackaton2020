import telebot
from telebot import types
from telebot import apihelper
token='970309668:AAHZX5VTRWBDhZ6XUyNKIJP9W28dVOqGtjA'
apihelper.proxy = {'https': 'socks5h://geek:socks@t.geekclass.ru:7777'}
bot = telebot.TeleBot(token=token)
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
        if call.data == "send_image":
            msg=bot.send_message(call.message.chat.id, "Send me a photo")
            bot.register_next_step_handler(callback=handle_add_photo, message=msg)
        if call.data == "recognize_image":
            bot.send_message(call.message.chat.id, "already")
@bot.message_handler(content_types=['photo'])
def handle_add_photo(message):
    file_info = bot.get_file(message.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    ind = predict(downloaded_file)
    dirr = str(ind)+'.txt'
    file=open(dirr,'r')
    text=file.read()
    bot.send_message(message.chat.id, text)
bot.polling(none_stop=True)
