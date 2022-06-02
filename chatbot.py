import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import tflearn, random, json
import tensorflow as tf
import numpy as np
import pickle

stemmer = LancasterStemmer()
responses = open("responses.json")
with responses as file:
    responses_loaded = json.load(file)
try:
    with open("data.pickle","rb") as data:
        words,classes,training,output = pickle.load(data)
except:
    words = []
    classes = []
    extra_info_x = []
    extra_info_y = []
    for intent in responses_loaded["intents"]:
        for patterns in intent["patterns"]:
            questions = nltk.word_tokenize(patterns)
            words.extend(questions)
            extra_info_x.append(patterns)
            extra_info_y.append(intent["tag"])
            # print(questions)
        if intent["tag"] not in classes:
                classes.append(intent["tag"])
        # print(classes)
    # building an array with certain conditions for x elements
    # sorted() is O(n) time complexity
    # set() has an O(1) lookup time vs. list() which has O(n) lookup time
    words = sorted(list(set([stemmer.stem(elements.lower()) for elements in words if "?" not in elements])))
    classes.sort()
    training = []
    output = []
    empty = [0 for _ in range(len(classes))]
    for x, extra_info in enumerate(extra_info_x):
        # bow is a bag of words array which stores true or false based on the given conditions
        bow = []
        questions = [stemmer.stem(elements) for elements in extra_info]
        for element in words:
            if element in questions:
                bow.append(1)
            else:
                bow.append(0)
        # [:] treats empty as a constant and does not allow for any side effects to happen as opposed to making emptyrow = empty
        emptyrow = empty[:]
        emptyrow[classes.index(extra_info_y[x])] = 1
        training.append(bow)
        output.append(emptyrow)
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as data:
        pickle.dump((words,classes,training,output),data)

tf.compat.v1.reset_default_graph()
# designing the structure of the neural network
neural_net = tflearn.input_data(shape=[None, len(training[0])])
neural_net = tflearn.fully_connected(neural_net,8)
neural_net = tflearn.fully_connected(neural_net,8)
neural_net = tflearn.fully_connected(neural_net,len(output[0]),activation="softmax")
neural_net = tflearn.regression(neural_net)
# defining the model
model = tflearn.DNN(neural_net)

#try:

#except:
model.fit(training,output,n_epoch=2000,batch_size=8,show_metric=True)
model.save("chatbot_model.tflearn")
model.load("chatbot_model.tflearn")

# intitialize and return bag of words
def bagofwords(string,words):
    bag = [0 for _ in range(len(words))]
    string_words = nltk.word_tokenize(string)
    string_words = [stemmer.stem(elements) for elements in string_words]
    for elem in string_words:
        for i,word in enumerate(words):
            if word == elem:
                bag[i] = 1
    return np.array(bag)

# chatting interface function
def chatting():
    print("Type something to ask the bot, and type \"end\" to stop chatting.")
    while True:
        userinput = input("You: ")
        if userinput.lower() == "end":
            break
        bot_output = model.predict([bagofwords(userinput,words)])
        bot_output_max_index = np.argmax(bot_output)
        tag = classes[bot_output_max_index]
        for tags in responses_loaded["intents"]:
            if tags["tag"] == tag:
                responses = tags["responses"]
        print(random.choice(responses))
chatting()