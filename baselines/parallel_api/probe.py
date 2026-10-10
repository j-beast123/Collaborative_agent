"""Read-only OpenAI authentication/model visibility check; no completion requests."""
import json
import urllib.error
import urllib.request
from baselines.parallel_api.credentials import read_api_key


def main():
    key=read_api_key(variable='OPENAI_API_KEY')
    request=urllib.request.Request('https://api.openai.com/v1/models/gpt-5.6-luna',
        headers={'Authorization':'Bearer '+key},method='GET')
    try:
        with urllib.request.urlopen(request,timeout=30) as response:body=json.load(response)
    except urllib.error.HTTPError as exc:
        raise SystemExit('OpenAI model visibility check failed: HTTP '+str(exc.code)) from None
    except OSError:
        raise SystemExit('OpenAI model visibility check failed: network/timeout error') from None
    if body.get('id')!='gpt-5.6-luna':raise SystemExit('OpenAI returned an unexpected model ID')
    print(json.dumps({'provider':'openai','model':body['id'],'visible':True,
        'completion_requested':False,'concurrency_capacity_verified':False}))


if __name__=='__main__':main()
