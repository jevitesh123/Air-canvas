import urllib.request, pathlib

url = 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task'
print('Downloading from', url)
path = pathlib.Path('models')
path.mkdir(exist_ok=True)
outfile = path/'hand_landmarker.task'
try:
    urllib.request.urlretrieve(url, outfile)
    print('Downloaded to', outfile, 'size', outfile.stat().st_size)
except Exception as e:
    print('Download failed:', e)
