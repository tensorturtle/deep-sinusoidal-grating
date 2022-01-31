import requests

def test_up(url='https://sine.tensorturtle.com'):
    r = requests.get(url)
    print(r.text)
    assert r.status_code == 200

if __name__ == "__main__":
    test_up('https://sine.tensorturtle.com')

