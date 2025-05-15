from yandex_music import Client
import json
import requests
from tqdm import tqdm

# Set session ID from Yandex Music (Required Yandex Plus)
SESSION_ID = 'y0_AgAAAABBBGWgAAG8XgAAAADfocBAD7JxjZEdRXWMa4a8w1BbQlHxIPY'

client = Client(SESSION_ID).init()

search_result = client.search('Pyrokinesis', type_='artist')
artist = search_result.artists.results[0]

tracks = []
for track in tqdm(client.artists_tracks(artist.id, page_size=100), desc="Парсинг треков"):
    track_data = {
        'title': track.title if track.title else '-',
        'album': track.albums[0].title if track.albums else '-',
        'release_year': track.albums[0].year if track.albums else '-',
        'lyrics': requests.get(track.get_lyrics()['download_url'])
                                   .text
                                   .replace('\n',' ')
                                   .replace('\r',' ')
                                   if track.get_lyrics() else '-',
    }
    tracks.append(track_data)

with open('./data/raw/pyrokinesis_raw.json', 'w', encoding='utf-8') as f:
    json.dump(tracks, f, ensure_ascii=False, indent=4)

print(f'Сохранено {len(tracks)} треков исполнителя {artist.name} в pyrokinesis_raw.json')
