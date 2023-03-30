import discord
from discord import app_commands
from discord.ext import commands
from keep_alive import keep_alive
import os

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

bot = commands.Bot(command_prefix=">", intents = discord.Intents.all())
TOKEN = ""#Token hidden for security reasons.

with open("intents.json") as file:
  data = json.load(file)

try:
  with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)
except:
  words = []
  labels = []
  docs_x = []
  docs_y = []


  for intent in data["intents"]:
    for pattern in intent["patterns"]:
      wrds = nltk.word_tokenize(pattern)
      words.extend(wrds)
      docs_x.append(wrds)
      docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
      labels.append(intent["tag"])

  words = [stemmer.stem(w.lower()) for w in words if w != "?"]
  words = sorted(list(set(words)))

  labels = sorted(labels)

  training = []
  output = []

  out_empty = [0 for _ in range(len(labels))]

  for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
      if w in wrds:
        bag.append(1)
      else:
        bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

  training = numpy.array(training)
  output = numpy.array(output)
  with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
  model.load("model.tflearn")
except:
  model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
  model.save("model.tflearn")


def bag_of_words(s, words):
  bag = [0 for _ in range(len(words))]
  s_words = nltk.word_tokenize(s)
  s_words = [stemmer.stem(word.lower()) for word in s_words]
  for se in s_words:
    for i, w in enumerate(words):
      if w == se:
        bag[i] = 1
  return numpy.array(bag)


async def on_ready():
  print("Bot is ready")
  try:
    synced = await bot.tree.sync()
    print(f"Synced {len(synced)} commands")
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.listening, name="to "+ str(len(bot.guilds)) + " servers!"))
  except Exception as e:
    print(e)
@bot.tree.command(name = "help")
async def help(interaction: discord.Interaction):
  await interaction.response.send_message("**TensorGPT**\nI am an intelligent AI chatbot, made by Tanmay G. Currently, my dataset is limited and inaccurate; but the plan is to improve it before the final release.\n*v0.1.2, built and modelled using TensorFlowLearnPY*\nInvite me to your servers and talk to me!\nMake requests to expand the dataset and add features using \"/request!\"\n**Need help? Join the support server:** https://discord.gg/aMCf3k5XTu\n*In addition, if a reply is wrong(like most will be), please don't report that in the discord server's #bug-reports channel. *\n*Note:This bot currently does not support legacy commands, only slash commands at this time.*\n\n**Commands**\n`/chat` - *Initiates the completely unique and original tensorflow model, made specifically for this bot. It can currently have small, predictable conversations.*\n`/request` - *Gives you a form to add features and replies for the AI chat command. It helps us expand our data.*\n`/help` - *You can always use it to recieve this same message!*\n\n**It takes a lot of work to be training this model, and requries hundreds of thousands, at times millions of data pieces. Your help is greatly appreciated, either through the request command or through directly becoming a developer.**\n\n*Buttons and an embed for easy usage coming soon*\n\n*Check the status page here: https://stats.uptimerobot.com/VnnNvU26k7*")

@bot.tree.command(name = "request")
async def request(interaction: discord.Interaction):
  await interaction.response.send_message("Want a feature or reply not currently implemented? Not to fear, we can add it to the dataset.\nForm: https://forms.gle/w5CtKqDGXZ1isSGf9")

@bot.tree.command(name = "ping")
async def ping(interaction: discord.Interaction):
  await interaction.response.send_message("**TensorGPT is online!**\nUse `/help` to get information on the bot!\n`Reaction time`: `{round(bot.latency*1000, 1)} ms`\n`Guild Count`: `{str(len(bot.guilds))} \n*Check the uptime of the bot:* https://stats.uptimerobot.com/VnnNvU26k7\n*Check individual shards:*\n`Main Shard (4 commands)`: https://stats.uptimerobot.com/VnnNvU26k7/788264503\n`GPT Shard (1 command)`: Coming soon.\n\nJoin the support server for latest updates on downtime: https://discord.gg/aMCf3k5XTu")

@bot.tree.command(name = "ask")
@app_commands.describe(arg = "Prompt")
async def ask(interaction: discord.Interaction, arg:str):
  def chat():
    while True:
      inp = arg
      results = model.predict([bag_of_words(inp, words)])[0]
      results_index = numpy.argmax(results)
      tag = labels[results_index]
      for tg in data["intents"]:
        if tg['tag'] == tag:
          responses = tg['responses']
      if results[results_index]<0.70:
        print(results[results_index])
        print("Not certain")
        return "Tensor found no suitable answer to that question! It was less than 70% sure about it. Please do help us out by submitting sample questions and answers by using `/request`."
      if random.choice(responses):
        print(results[results_index])
        return random.choice(responses)
      else:
        return "Tensor found no suitable answer to that question! Please report the question or statement, so that it can be considered to be added to the training dataset for the next update. Join the support server if you have questions with the help command or through my about me."
  await interaction.response.send_message(chat())    

keep_alive()
try:
  bot.run(TOKEN)
except:
  os.system("kill 1")
