import imageio
import os 
from glob import glob 

# Ask for options
optionList = '''
Please enter the image filtering string, e.g. "*plot.png"
'''

def make_gif(image_list, output_name, fps):
    images = []
    image_list.sort()
    for filename in image_list:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_name, images, fps)

if __name__=='__main__':
    # Ask for file info
    path = input('input file directory: ')
    if not os.path.exists(path):
        raise ValueError('')
    
    print(optionList)
    option = input('')

    output_path = os.path.join(path, 'movie.gif')
    outfile = input(f'save the file to: (default {output_path})')
    if len(outfile < 1):
        outfile = output_path
    filelist = glob(path)

    fps = input('frames per second (default 1)')
    if len(fps) < 1:
        fps = 1
    make_gif(filelist, outfile, fps)

