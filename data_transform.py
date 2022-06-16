# Author: Bozitao Zhong
# Date: 2022-06-17
# This program is to transform the svg free-hand sketch to 28*28 pixel images
# some functions are from https://github.com/CMACH508/RPCL-pix2seq


import numpy as np
import svgwrite
import cairosvg
import os



def get_bounds(data):
    # Return bounds of data.
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0])
        y = float(data[i, 1])
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)



def draw_strokes(data, svg_filename='sample.svg', width=48, margin=1.5, color='black'):
    # convert sequence data to svg format
    min_x, max_x, min_y, max_y = get_bounds(data)
    if max_x - min_x > max_y - min_y:
        norm = max_x - min_x
        border_y = (norm - (max_y - min_y)) * 0.5
        border_x = 0
    else:
        norm = max_y - min_y
        border_x = (norm - (max_x - min_x)) * 0.5
        border_y = 0
  
    # normalize data
    norm = max(norm, 10e-6)
    scale = (width - 2*margin) / norm
    dx = 0 - min_x + border_x
    dy = 0 - min_y + border_y
  
    abs_x = (0 + dx) * scale + margin
    abs_y = (0 + dy) * scale + margin
  
    # start converting
    dwg = svgwrite.Drawing(svg_filename, size=(width,width))
    dwg.add(dwg.rect(insert=(0, 0), size=(width,width),fill='white'))
    lift_pen = 1
    p = "M%s,%s " % (abs_x, abs_y)
    command = "m"
    for i in range(len(data)):
        if (lift_pen == 1):
            command = "m"
        elif (command != "l"):
            command = "l"
        else:
            command = ""
        x = float(data[i,0]) * scale
        y = float(data[i,1]) * scale
        lift_pen = data[i, 2]
        p += command+str(x)+","+str(y)+" "
    the_color = color  # "black"
    stroke_width = 1
    dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
    dwg.save()



if __name__ == '__main__':
    # important presets
    npz_file_location = "./dataset/"
    pixel_file_location = "./dataset_pixel/"
    pixel_width = 28


    # iterate for each dataset
    npz_file_list = os.listdir(npz_file_location)
    class_list = [npz_file_name.split('.')[0][10:] for npz_file_name in npz_file_list]
    # or chose one of them (annotate this to skip)
    npz_file_list = [npz_file_list[0]]
    class_list = [class_list[0]]


    for npz_file_name, class_name in zip(npz_file_list, class_list):
        # load file first
        npz_file_path = npz_file_location + npz_file_name
        npz_file = np.load(npz_file_path,allow_pickle=True,encoding='latin1')

        # cut into train, test, valid
        for tag in ["train","test","valid"]:
            data = npz_file[tag]

            # create directory
            pixel_file_path = pixel_file_location + class_name + "/" + tag
            os.makedirs(pixel_file_path)

            # iterate for each stroke
            for i in range(len(data)):
                svg_file_path = pixel_file_path + "/" + str(i) + ".svg"
                png_file_path = pixel_file_path + "/" + str(i) + ".png"

                # draw stroke
                stroke_data = data[i]
                draw_strokes(stroke_data,
                             svg_filename=svg_file_path,
                             width=pixel_width)

                # convert svg to png
                cairosvg.svg2png(url=svg_file_path, write_to=png_file_path)
                
                os.remove(svg_file_path)
        print(class_name+' done')


