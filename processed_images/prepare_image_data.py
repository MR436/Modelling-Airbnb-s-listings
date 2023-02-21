from PIL import Image
import os
import math

path = '../images'
filelist = []

all_images_path = []
# print(os.listdir(path))

def resize_images():
    for root, folders, files in os.walk(path):
        #print(files)
        for file in files:
            filelist.append(os.path.join(root, file))
        #print(filelist)
    size_list = []
    
    for file in filelist:
        
        image = Image.open(file)
        w,h = image.size
        #print(w,h)
        size_list.append(h)
        aspect_ratio = h/w     
        new_size = (math.floor(w*aspect_ratio), 156)
        print(new_size)
        try: 
            resized_images = image.resize(new_size)
            #print(resized_images)
        except OSError:
            continue

        #print(min(size_list))
        #print(image.mode)
        #print(file)
        image_path = file.split('\\')[2]
        #all_images_path.append(image_path)
        #print(all_images_path)

        
        resized_images.save(f'./{image_path}')

        #print(min(size_list))
    
if __name__ == "__main__":
    resize_images()




