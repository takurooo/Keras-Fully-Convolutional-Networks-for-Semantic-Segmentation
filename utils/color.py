#-------------------------------------------
# import
#-------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------------------
# defines
#-----------------------------------------

#-----------------------------------------
# functions
#-----------------------------------------


def make_cmap():
    ctbl = ((0x80, 0, 0), (0, 0x80, 0), (0, 0, 0x80),
            (0x40, 0, 0), (0, 0x40, 0), (0, 0, 0x40),
            (0x20, 0, 0), (0, 0x20, 0))

    n = 256
    lookup = np.zeros((n, 3)).astype(np.int32)
    for i in range(0, n):
        r, g, b = 0, 0, 0
        for j in range(0, 7):
            bit = (i >> j) & 1
            if bit:
                r |= ctbl[j][0]
                g |= ctbl[j][1]
                b |= ctbl[j][2]

        lookup[i, 0], lookup[i, 1], lookup[i, 2] = r, g, b
    return lookup[0:21]


#-----------------------------------------
# main
#-----------------------------------------
if __name__ == '__main__':
    cmap = make_cmap()

    for idx in range(0, 21):
        x = np.ones((128, 128)) * idx

        row, col = x.shape
        dst = np.ones((row, col, 3))
        for i in range(21):
            dst[x == i] = cmap[i]

        dst = np.uint8(dst)
        plt.subplot(5, 5, idx + 1)
        plt.xticks(color="None")
        plt.yticks(color="None")
        plt.tick_params(length=0)
        plt.imshow(dst)

        #img = Image.fromarray(dst)
        # img.save("color_map_test.png")

    plt.show()
