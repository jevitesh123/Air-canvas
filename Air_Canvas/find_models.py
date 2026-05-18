import pathlib, sys
p = pathlib.Path(sys.prefix) / 'Lib' / 'site-packages'
models = [x for x in p.rglob('*hand*') if x.suffix in ('.tflite', '.pb')]
print('found', len(models), 'models')
for m in models[:20]:
    print(m)
