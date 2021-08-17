from PIL import Image
from numpy import array
from pathlib import Path
import numpy as np
import os
import random
import math

# List of cities in 'test' set
test = ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']

# List of cities in 'train' set
train = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 
         'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']

# List of cities in 'val' set
val = ['frankfurt', 'lindau', 'munster']

# 34 classes of objects
name = ['unlabeled', 'ego_vehicle', 'rectification_border', 'out_of_roi', 'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking', 'rail_track', 
        'building', 'wall', 'fence','guard_rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 
        'sky', 'person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']

# Classes of anomalies (add items to this list if you want other classes of objects)
object_class = ['bicycle', 'bus', 'car', 'fence', 'motorcycle', 'person', 'pole', 'traffic_light', 'traffic_sign', 'trailer', 'train', 'truck', 'vegetation']


# Helper function for createCandidates()
# Extract an object in an image
# Save object image to 'objectImg/class_of_object/city_number_instance_id.png'
# Return False if the object does not exist or width/length < 300 pixels
def extractObject(fine_image, source_image, id, instance_id, city, number):
    fine_pixels = array(fine_image)

    if id <= 23:
        object_id = id
    else:
        object_id = id * 1000 + instance_id

    if object_id not in fine_pixels:
        return False

    # Change the value of the foreground to be 1 and background to be 0
    if id == 0:
        fine_pixels = np.where(fine_pixels!=object_id, -2, fine_pixels)
        fine_pixels = np.where(fine_pixels==object_id, 1, fine_pixels)
        fine_pixels = np.where(fine_pixels==-2, 0, fine_pixels)
    else:
        fine_pixels = np.where(fine_pixels!=object_id, 0, fine_pixels)
        fine_pixels = np.where(fine_pixels==object_id, 1, fine_pixels)

    # Find the boundary of the object
    positions = np.nonzero(fine_pixels)
    top = min(positions[0])
    bot = max(positions[0])
    left = min(positions[1])
    right = max(positions[1])

    if left >= right or top >= bot or (right-left < 300 and bot-top < 300):
        return True

    object_pixels = array(source_image)

    # Apply the fine_pixels as a mask to the source image (multiply every coresponding pixels)
    for i in range(3):
        object_pixels[:, :, i] = object_pixels[:, :, i] * fine_pixels

    # Crop to fit
    object_image = Image.fromarray(object_pixels)
    object_image = object_image.crop((left, top, right, bot))
    
    object_path = 'objectImg/' + name[id] + '/'

    if not Path(object_path).exists():
        Path(object_path).mkdir()
    
    object_path += city + '_' + number + '_' + str(instance_id) + '.png'

    object_image.save(object_path)
    print('\t' + object_path)
    return True

# Go through all the images in test/train/val set and extract all the objects
# If you want to use  the images in a different set, make sure you change the paths
def createCandidates():
    if not Path('objectImg/').exists():
        Path('objectImg/').mkdir()

    for city in val:
        temp_path = 'leftImg8bit_trainvaltest/leftImg8bit/val/' + city + '/'
        for index,path in enumerate(Path(temp_path).iterdir()):
            temp = str(path.stem)
            number = temp[temp.index('0'):temp.index('0')+13]
            fine_path = 'gtFine_trainvaltest/gtFine/val/' + city + '/' + city + '_' + number + '_gtFine_instanceIds.png'
            source_path = 'leftImg8bit_trainvaltest/leftImg8bit/val/' + city + '/' + city + '_' + number + '_leftImg8bit.png'

            print('Start working on #' + str(index) + ': ' + city + '_' + number + '...')
            fine_image = Image.open(fine_path)
            source_image = Image.open(source_path)

            for id in range(len(name)):
                instance_id = 0

                # Classes with id between 0-23 do not have an instance_id
                if id >= 24 and id <= 33:
                    while (extractObject(fine_image, source_image, id, instance_id, city, number)):
                        instance_id += 1
                else:
                    extractObject(fine_image, source_image, id, instance_id, city, number)

            fine_image.close()
            source_image.close()
            
            print('Done!\n')

# Add objects (anomalies) in 'objectImg/' to the background images in 'backgroundImg/'
# Save modified image to 'modifiedImg/city_number_class_of_object.png'
def addAnomaly():
    if not Path('modifiedImg/').exists():
        Path('modifiedImg/').mkdir()

    background_path = list(Path('backgroundImg/').iterdir())
    counter = 0
    total_background = len(background_path)

    while counter < total_background:
        for oc in object_class:
            for anomaly_path in Path('objectImg/' + oc + '/').iterdir():
                print('Start generating #' + str(counter) + ':', end=' ')
                temp_str = str(background_path[counter])
                city = temp_str[temp_str.index('\\')+1:temp_str.index('_')]
                number = temp_str[temp_str.index('_'):temp_str.index('_')+15]
                save_path = 'modifiedImg/' + city + number + oc + '.png'
                print(save_path + '...', end='')
                
                anomaly_image = Image.open(anomaly_path)
                background_image = Image.open(background_path[counter])

                # Randomly rotate the anomaly
                angle = random.randint(0, 360)
                rotated_image = anomaly_image.rotate(angle, expand=True)

                # Crop to fit
                positions = np.nonzero(rotated_image)
                top = min(positions[0])
                bot = max(positions[0])
                left = min(positions[1])
                right = max(positions[1])
                rotated_image = rotated_image.crop((left, top, right, bot))

                # Resize the anomaly if it is too big (anomaly occupies less than 50% of the background image)
                width, height = rotated_image.size
        
                if width > 1024:
                    r = 1024 / width
                    width *= r
                    height *= r
                    if height > 512:
                        r = 512 / height
                        width *= r
                        height *= r
                    newsize = (int(width), int(height))
                    rotated_image = rotated_image.resize(newsize)
                elif height > 1024:
                    r = 1024 / height
                    width *= r
                    height *= r
                    if width > 512:
                        r = 1024 / width
                        width *= r
                        height *= r
                    newsize = (int(width), int(height))
                    rotated_image = rotated_image.resize(newsize)

                mask_pixels = array(rotated_image.convert('L'))
                mask_pixels = np.where(mask_pixels!=0, 255, mask_pixels)
                mask_image = Image.fromarray(mask_pixels)

                # Randomly place the anomaly in the center square of the background image
                x = random.randint(512, 1536-rotated_image.width)
                y = random.randint(0, 1024-rotated_image.height)
                background_image.paste(rotated_image, (x, y), mask_image)

                background_image.save(save_path)

                anomaly_image.close()
                background_image.close()
                rotated_image.close()
                mask_image.close()
                counter +=1
                print('Done!')

                if counter == total_background:
                    return

# Count the number of objects in each class, the total number of objects, and the total number of background images
# Sort object_class by the number of objects it contains
def sortObjectClass():
    background = list(Path('backgroundImg/').iterdir())
    print('Total number of background images:' , len(background))

    global object_class
    sum = 0
    sorted_class = []
    sorted_count = []
    for oc in object_class:
        object =  list(Path('objectImg/' + oc + '/').iterdir())
        count = len(object)
        if len(sorted_count) == 0:
            sorted_count.append(count)
            sorted_class.append(oc)
        else:
            index = 0
            while index < len(sorted_count) and count >= sorted_count[index]:
                index += 1
            sorted_count.insert(index, count)
            sorted_class.insert(index, oc)
        sum += count
    print('\nTotal number of objects:' , sum)

    object_class = sorted_class
    for i in range(len(object_class)):
        print(object_class[i] + ":", sorted_count[i])



def main():
    sortObjectClass()
    addAnomaly()
    #createCandidates()

if __name__ == '__main__':
    main()