import os

def rename_files_in_folder(folder_path, start_num):
    files = os.listdir(folder_path)

    jpg_files = sorted([f for f in files if f.endswith('.jpg')])
    json_files = sorted([f for f in files if f.endswith('.json')])

    if len(jpg_files) != len(json_files):
        print("error")
        return

    for i, (jpg_file, json_file) in enumerate(zip(jpg_files, json_files)):
        new_jpg_name = f"frame_{start_num + i}.jpg"
        new_json_name = f"frame_{start_num + i}.json"

        old_jpg_path = os.path.join(folder_path, jpg_file)
        new_jpg_path = os.path.join(folder_path, new_jpg_name)
        old_json_path = os.path.join(folder_path, json_file)
        new_json_path = os.path.join(folder_path, new_json_name)

        os.rename(old_jpg_path, new_jpg_path)
        os.rename(old_json_path, new_json_path)
#
folder_path = '/home/junyan/Documents/sample_img/sample_img25/'
start_num = 3596
rename_files_in_folder(folder_path, start_num)
