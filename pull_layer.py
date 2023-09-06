import os
import shutil

new_dir = "uhem_trained"

base_dir = "logs"

models = os.listdir(base_dir)

for m in models:
    # if m!="dgcnn":continue
    model_path = os.path.join(base_dir, m)
    model_path2  = os.path.join(new_dir, m)
    
    layers = os.listdir(model_path)
    layers = sorted(layers)
    
    for l in layers:
        layer_path = os.path.join(model_path, l)
        layer_path2 = os.path.join(model_path2, l)
        dates = os.listdir(layer_path)
        
        for d in dates:
            dates_path = os.path.join(layer_path, d)
            dates_path2 = os.path.join(layer_path2, d)
            files = os.listdir(dates_path)
            files = sorted(files)
            print(dates_path, files[0])
            if "log.txt" in files:
                print(dates_path, "log.txt")
            
            file = os.path.join(dates_path, files[0])
            log = os.path.join(dates_path, "log.txt")
            event = os.path.join(dates_path, files[-2])
            
            file2 = os.path.join(dates_path2, files[0])
            log2 = os.path.join(dates_path2, "log.txt")
            event2 = os.path.join(dates_path2, files[-2])
            
            
            os.makedirs(dates_path2, exist_ok=True)
            shutil.copyfile(file, file2)
            shutil.copyfile(log, log2)
            shutil.copyfile(event, event2)
            
