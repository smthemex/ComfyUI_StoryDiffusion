import re


def get_phrase_idx(tokenizer, phrase, prompt, get_last_word=False, num=0):
    def is_equal_words(pr_words, ph_words):
        if len(pr_words) != len(ph_words):
            return False
        for pr_word, ph_word in zip(pr_words, ph_words):
            if "-"+ph_word not in pr_word and ph_word != re.sub(r'[.!?,:]$', '', pr_word):
                return False
        return True

    phrase_words = phrase.split()
    if len(phrase_words) == 0:
        return [0, 0], None
    if get_last_word:
        phrase_words = phrase_words[-1:]
    # prompt_words = re.findall(r'\b[\w\'-]+\b', prompt)
    prompt_words = prompt.split()
    start = 1
    end = 0
    res_words = phrase_words
    for i in range(len(prompt_words)):
        if is_equal_words(prompt_words[i:i+len(phrase_words)], phrase_words):
            if num != 0:
                # skip this one
                num -= 1
                continue
            end = start
            res_words = prompt_words[i:i+len(phrase_words)]
            res_words = [re.sub(r'[.!?,:]$', '', w) for w in res_words]
            prompt_words[i+len(phrase_words)-1] = res_words[-1]  # remove the last punctuation
            for j in range(i, i+len(phrase_words)):
                end += len(tokenizer.encode(prompt_words[j])) - 2
            break
        else:
            start += len(tokenizer.encode(prompt_words[i])) - 2

    if end == 0:
        return [0, 0], None

    return [start, end], res_words


def get_eot_idx(tokenizer, prompt):
    words = prompt.split()
    start = 1
    for w in words:
        start += len(tokenizer.encode(w)) - 2
    return start
