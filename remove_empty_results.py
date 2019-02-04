import os
import shutil

empty_folders = []

for folder in os.listdir('results'):
    rel_path = os.path.join('results',folder)
    files = list(os.listdir(rel_path))
    types = {f.rsplit('.',1)[-1] for f in files}
    if len(files) < 4:
        empty_folders.append(rel_path)
        print("-",folder,"empty:",len(files),"files with types",types)

if len(empty_folders) > 0:
    print("\nFound {} unfinished folders. Delete (y/n)?".format(len(empty_folders)))
    response = input()

    if response == 'y':
        for folder in empty_folders:
            shutil.rmtree(folder)
        print('- Deleted {} folders.'.format(len(empty_folders)))
else:
    print("\nFound no unfinished folders.")