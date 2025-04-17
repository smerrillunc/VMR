#!/usr/bin/env python

import yt_dlp
import os
import gc
import random
import time
import subprocess

import random
import requests
import subprocess
import time

import tqdm
import argparse
from functools import wraps

def load_proxies(file_path='proxies.txt'):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def is_proxy_alive(proxy):
    try:
        response = requests.get(
            'https://httpbin.org/ip',
            proxies={'http': proxy, 'https': proxy},
            timeout=5
        )
        if response.status_code == 200:
            print(f"✅ Proxy works: {response.json()['origin']}")
            return True
    except:
        pass
    print(f"❌ Dead proxy: {proxy}")
    return False
def retry(times=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"⚠️ Attempt {i+1} failed: {e}")
                    time.sleep(delay)
            print("❌ All attempts failed.")
            return []
        return wrapper
    return decorator


@retry(times=3, delay=2)
def fetch_proxies_proxyscrape():
    print("🌐 Fetching US HTTP & SOCKS proxies from ProxyScrape...")
    proxies = []

    base_url = "https://api.proxyscrape.com/v2/?request=getproxies&timeout=3000&country=us&ssl=all&anonymity=all"

    types = {
        "http": f"{base_url}&protocol=http",
        "socks4": f"{base_url}&protocol=socks4",
        "socks5": f"{base_url}&protocol=socks5"
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    for proto, url in types.items():
        try:
            time.sleep(1.5)  # Throttle to avoid 429
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                print(f"🚫 ProxyScrape ({proto}) says: Too Many Requests. Skipping...")
                continue
            response.raise_for_status()
            new_proxies = [f"{proto}://{line.strip()}" for line in response.text.splitlines() if line.strip()]
            proxies.extend(new_proxies)
        except Exception as e:
            print(f"❌ ProxyScrape ({proto}) failed: {e}")
    
    return proxies

@retry(times=3, delay=2)
def fetch_proxies_geonode():
    print("🌐 Fetching US HTTP proxies from GeoNode...")
    url = "https://proxylist.geonode.com/api/proxy-list?limit=50&page=1&sort_by=lastChecked&sort_type=desc&country=US&protocols=http"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        proxies = [f"http://{proxy['ip']}:{proxy['port']}" for proxy in data['data']]
        return proxies
    except Exception as e:
        print(f"❌ GeoNode failed: {e}")
        return []
    
@retry(times=3, delay=2)
def fetch_proxies_proxylist_download():
    print("🌐 Fetching US HTTP proxies from proxy-list.download...")
    url = "https://www.proxy-list.download/api/v1/get?type=http&country=US"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        proxies = [f"http://{line.strip()}" for line in response.text.splitlines() if line.strip()]
        return proxies
    except Exception as e:
        print(f"❌ proxy-list.download failed: {e}")
        return []

def fetch_all_us_proxies():
    proxies = set()
    sources = [
        fetch_proxies_proxyscrape,
        fetch_proxies_geonode,
        fetch_proxies_proxylist_download
    ]
    for fetch_func in sources:
        proxies.update(fetch_func())

    # Final filter: Only include HTTP and SOCKS proxies
    valid_prefixes = ("http://", "socks4://", "socks5://")
    filtered = [p for p in proxies if p.startswith(valid_prefixes)]

    print(f"✅ Total valid HTTP/SOCKS US proxies fetched: {len(filtered)}")
    return filtered


def is_youtube_accessible(proxy):
    test_url = "https://www.youtube.com"
    try:
        resp = requests.get(test_url, proxies={"http": proxy, "https": proxy}, timeout=5)
        return resp.status_code == 200
    except:
        return False

def download_video(video_url, proxy):
    print(f"📥 Attempting download with {proxy}")
    try:
        subprocess.run([
            "yt-dlp",
            "--proxy", proxy,
            "--user-agent", "Mozilla/5.0",
            "--limit-rate", "500K",
            "--no-check-certificate",
            "-o", "/work/users/s/m/smerrill/Youtube8m/%(width&video|audio)s/%(id)s.%(ext)s",
            "--postprocessor-args", "-ss 00:00:00 -t 360",
            '--retries', '3',
            video_url
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ yt-dlp failed with proxy: {proxy}")
        return False

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')

  parser.add_argument("-s", "--start_index", type=int, default=0, help='YoutubeID Index to start on in YoutubeID File')
  parser.add_argument("-e", "--end_index", type=int, default=100, help='YoutubeID Index to end on in YoutubeID File')
  parser.add_argument("-p", "--path", type=str, default='/work/users/s/m/smerrill/Youtube8m', help='Path to YoutubeID file.  This will also be where output featuers are saved')
  args = vars(parser.parse_args())

 
  os.makedirs(args['path'] + '/video', exist_ok=True)
  os.makedirs(args['path'] + '/audio', exist_ok=True)

  downloaded_vids = os.listdir(args['path'] + '/audio')
  downloaded_vids = [x.split('.')[0] for x in downloaded_vids]

  # Here are the youtube ids used by original VM-NET
  with open(args['path'] + '/Youtube_ID.txt', 'r') as file:
      content = file.read()  # Read the entire content of the file
  youtube_urls = content.split('\n')

  youtube_urls = youtube_urls[args['start_index']:args['end_index']]

  all_proxies = []
  i = 0
  for video_url in tqdm.tqdm(youtube_urls):
      # video id      
      vid = video_url.split('=')[1]

      if vid in downloaded_vids:
        print(f"VID: {vid} alread processed, skipping")
        continue


      print(f"processing VID: {vid}")
      if i >= len(all_proxies):
        i = 0
        all_proxies = fetch_all_us_proxies()

      random.shuffle(all_proxies)

      success = False

      while i < len(all_proxies):
        proxy = all_proxies[i]
        if not proxy.startswith(("http://", "socks4://", "socks5://")):
            continue

        if is_youtube_accessible(proxy):
            success = download_video(video_url, proxy)
            if success:
                print("✅ SUCCESS — Video downloaded.")
                break
        else:
            print(f"⚠️ Proxy not allowed by YouTube: {proxy}")

        i += 1

      if not success:
          print("💀 All proxies failed or timed out. Continuing to next video...")
      
  print("Download Complete")
