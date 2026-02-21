import glob

from PIL import Image

if __name__=="__main__":

    frames = [Image.open(im) for im in sorted(glob.glob("visualizations/*.png"))]
    frame_one = frames[0]
    frame_one.save("visu.gif", format="GIF", append_images=frames, save_all=True, duration=300, loop=0)
