def set_open_params(
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
):
    """ set openai parameters"""
    openai_params = {'model': model, 'temperature': temperature, 'max_tokens': max_tokens, 'top_p': top_p,
                     'frequency_penalty': frequency_penalty, 'presence_penalty': presence_penalty}

    return openai_params
