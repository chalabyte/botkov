#!/usr/bin/env python3

import os, sys, re
import asyncio, random
import json, yaml
from ast import literal_eval

import discord
import numpy as np

VARIANCE = 2

BOT = None
CONFIG = {}

RESPONSE_CHANCE = 0
SHUTUP = False

client = discord.Client()

def sanitize(item):
    return item \
        .replace("🏼", "").replace("ê", "e").replace("à", "a") \
        .lower() \
        .strip()


class Bot(object):
    def __init__(self, brain):
        self.trans_prob             = brain["data"]
        self.k                      = brain["k"]
        self.target_sentence_length = brain["target_sentence_length"] + 2 # because of #START# and #END#

        self._should_finish         = False
    

    def get_word(self, direction, current_element):
        
        words = self.trans_prob[current_element][direction]
        if self._should_finish:
            if direction == "prev" and "#START#" in words:
                return "#START#"
            if direction == "next" and "#END#" in words:
                return "#END#"
        try:
            p=np.array(list(words.values()))
        except KeyError:
            return None
        p /= p.sum()

        if (len(list(words.keys())) == 0): return None

        return np.random.choice(
            list(words.keys()),
            p=p
        )

    def generate(self, original_word=None):


        # first find all keys having original word in it.
        if original_word is None:
            elements        = [tuples for tuples in self.trans_prob]
        else:
            elements        = [tuples for tuples in self.trans_prob if original_word in tuples]

        if len(elements) == 0: return None

        idx             = np.random.choice(len(elements))
        orig_element    = elements[idx]

        sentence = [elem for elem in orig_element[::-1]]

        randlist = np.array(list(range(self.k["max"]+1))[self.k["min"]:])
        probs = ( [x**VARIANCE for x in randlist] / np.array([x**VARIANCE for x in randlist]).sum())[::-1]

        # then, get some previous words from it.
        element = orig_element
        while True:
            if (len(sentence) >= self.target_sentence_length): self._should_finish = True
            try:
                previous_word = self.get_word("prev", element)
            except KeyError:
                element = element[:-1]
                if (len(element) == 0): break

                continue
            if (previous_word == "#START#" or previous_word is None): break
            sentence += [previous_word]
            if len(element) == self.k["max"]:
                newRandlist = np.array(list(range(self.k["max"]))[self.k["min"]:])
                newProbs = ( [x**VARIANCE for x in newRandlist] / np.array([x**VARIANCE for x in newRandlist]).sum())[::-1]
                element = tuple(sentence[-(np.random.choice(newRandlist, p=newProbs)):])[::-1]
            else:
                element = tuple(sentence[-(np.random.choice(randlist, p=probs)):])[::-1]
        sentence.reverse()

        # then, get some next words.
        element = orig_element
        while True:
            if (len(sentence) >= self.target_sentence_length): self._should_finish = True
            try:
                next_word = self.get_word("next", element)
            except KeyError:
                element = element[1:]
                if (len(element) == 0): break
                continue

            if (next_word == "#END#" or next_word is None): break
            sentence += [next_word]
            if len(element) == self.k["max"]:
                newRandlist = np.array(list(range(self.k["max"]))[self.k["min"]:])
                newProbs = ( [x**VARIANCE for x in newRandlist] / np.array([x**VARIANCE for x in newRandlist]).sum())[::-1]
                element = tuple(sentence[-(np.random.choice(newRandlist, p=newProbs)):])
            else:
                element = tuple(sentence[- (np.random.choice(randlist, p=probs)):])


        self._should_finish = False
        txt = " ".join([word for word in sentence if word not in ["#START#", "#END#"]])
        txt = re.sub(r'\s+', ' ', txt)
        txt = txt[0].upper() + txt[1:]
        txt = txt.replace(" . . . ", "... ")
        txt = txt.replace(" . ", ". ")
        txt = txt.replace(" .", ". ")
        txt = txt.replace(" , ", ", ")
        txt = txt.replace(" ( ", " (")
        txt = txt.replace(" ) ", ") ")
        txt = txt.replace(" ' ", "'")

        return txt



def main():
    global BOT, CONFIG, RESPONSE_CHANCE

    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        CONFIG = yaml.load(f.read(), Loader=yaml.FullLoader)

    brain = {}
    if os.path.isfile(CONFIG["core"]["brain_file"]):
        try:
            with open(CONFIG["core"]["brain_file"], "r") as f:
                brain = json.loads(f.read())
            brain["data"] = {literal_eval(k): v for k,v in brain["data"].items()}
            print(f"Loaded brain {CONFIG['core']['brain_file']} containing {len(list(brain['data'].keys()))} tokens.")
        except json.decoder.JSONDecodeError:
            exit(f"Error parsing {CONFIG['core']['brain_file']}.")
    else:
        exit(f"{CONFIG['core']['brain_file']} not found. create one using train.py")

    BOT = Bot(brain)

    if len(sys.argv) > 2 and sys.argv[2] == 'local':
        while True:
            inp = sanitize(input("> ")).split(" ")
            np.random.shuffle(inp)
            for word in inp:
                ret=BOT.generate(word)
                if ret is not None: break

            if ret is None:
                ret = BOT.generate()
            print(ret)

    elif len(sys.argv) > 2 and sys.argv[2] == 'genalot':
        for _ in range(1000):
            ret = BOT.generate()
            print(ret)
        return

    RESPONSE_CHANCE = CONFIG['bot']['response_chance']
    client.run(CONFIG["bot"]["token"])
    

##
## DISCORD STUFF
##

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    activity = discord.Activity(name=CONFIG["bot"]["activity"], type=discord.ActivityType.watching)
    await client.change_presence(status=discord.Status.online, activity=activity)


@client.event
async def on_message(message):
    global RESPONSE_CHANCE, SHUTUP
    if message.author.id == 198861674424303616:
        if message.content.startswith(f"{CONFIG['bot']['cmd']} rep"):
            args = message.content.split(" ")[2:]
            if len(args) == 0:
                await message.channel.send(f"La chance de réponse est à {RESPONSE_CHANCE}")
            else:
                newChanceResp = float(args[0])
                if newChanceResp < 0 or newChanceResp > 1:
                    await message.channel.send("La nouvelle chance de réponse ne peux pas être < 0 ou > 1")
                else:
                    RESPONSE_CHANCE = newChanceResp
                    await message.channel.send(f"La nouvelle chance de réponse est ä {RESPONSE_CHANCE}")
            return
        if message.content.startswith("!shutup"):
            SHUTUP = True
            await message.channel.send("Ok tocar je ferme ma gueule")
            return
        if message.content.startswith("!parle"):
            SHUTUP = False
            await message.channel.send("Ok boomer je parle")
            return

    if message.author == client.user:
        return
    if not client.is_ready:
        return
    if random.random() > RESPONSE_CHANCE:
        return
    if SHUTUP:
        return

    inp = sanitize(message.content).split(" ")
    np.random.shuffle(inp)
    for word in inp:
        ret=BOT.generate(word)
        if ret is not None: break


    if ret is None: return

    async with message.channel.typing():
        await asyncio.sleep(random.randint(1,3))
        await message.channel.send(ret)
        

if __name__=='__main__':
    main()