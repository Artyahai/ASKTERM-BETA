import requests

def translate(text, source, target):
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source}&tl={target}&dt=t&q={requests.utils.quote(text)}"
    try:
        detect = ''
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return ''.join([x[0] for x in response.json()[0]])
    except:
        pass


